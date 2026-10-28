/* Route 1 only: isolated single-thread L2 HNSW recorder/comparator.
 * Search control flow adapted from facebookresearch/faiss v1.8.0 HNSW.cpp,
 * Copyright Meta Platforms, Inc. and affiliates, MIT license. Uses the installed
 * FAISS distance computer, heap and result handler; never replaces that library.
 * No continuation controller or second-view search is implemented here.
 */
#include <faiss/IndexHNSW.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using faiss::idx_t;
constexpr int EW=14;
struct Counters { double* s; };

// Lookup-free inversion of the unchanged Tri threshold. This is a plug-in
// geometry gate, not a posterior probability or an adaptive-search certificate.
double critical_beta(double fcritical, double rho) {
    double c=std::sqrt((1-rho)*(1+rho))*(std::sqrt(fcritical)-1/std::sqrt(fcritical));
    double root=(std::hypot(c,2.0)+c)/2.0;
    return root*root;
}

struct Search {
    faiss::IndexHNSW* index;
    const float *query, *qp, *projected, *oracle;
    int m, k, ef, mode, rule, audit;
    double fc, cc;
    double *events, *s;
    int capacity, used=0;
    std::unique_ptr<faiss::DistanceComputer> dc;

    Search(faiss::IndexHNSW* ix,const float* q,const float* p,const float* xp,
           int dim,int outk,int beam,int md,int rl,double f,double c,int au,
           const float* oc,double* ev,int cap,double* stats)
        :index(ix),query(q),qp(p),projected(xp),oracle(oc),m(dim),k(outk),ef(beam),
         mode(md),rule(rl),audit(au),fc(f),cc(c),events(ev),s(stats),capacity(cap),
         dc(ix->storage->get_distance_computer()) {
        std::fill(s,s+20,0.0); dc->set_query(query);
    }
    float distance(int row,bool upper=false) {
        s[0]++; if(upper)s[1]++;
        return (*dc)(row);
    }
    float project(int row) {
        s[2]++; return faiss::fvec_L2sqr(qp,projected+size_t(row)*m,m);
    }
    void record(int x,int pivot,int ref,float d,double threshold,double sx,
                double sz,double rho,int safe,int predicted,int deferred,double result_threshold,double frontier_threshold) {
        if(!events)return;
        if(used>=capacity)throw std::runtime_error("event capacity exceeded");
        double* e=events+size_t(used++)*EW;
        double values[EW]={double(x),double(pivot),double(ref),d,threshold,sx,sz,rho,
                           double(safe),double(predicted),double(deferred),s[0],result_threshold,frontier_threshold};
        std::copy(values,values+EW,e);
    }
    void run(float* D,idx_t* I) {
        auto& h=index->hnsw;
        if(index->metric_type!=faiss::METRIC_L2 || h.upper_beam!=1 ||
           !h.search_bounded_queue || !h.check_relative_distance)
            throw std::runtime_error("unsupported HNSW contract");
        using RH=faiss::HeapBlockResultHandler<faiss::HNSW::C>;
        RH handler(1,D,I,k); RH::SingleResultHandler res(handler);res.begin(0);
        int nearest=h.entry_point;
        if(nearest<0)throw std::runtime_error("empty index");
        float dn=distance(nearest,true);
        for(int level=h.max_level;level>=1;--level) {
            for(;;) {
                int prev=nearest;size_t b,e;h.neighbor_range(nearest,level,&b,&e);
                for(size_t j=b;j<e && h.neighbors[j]>=0;++j) {
                    int v=h.neighbors[j];float d=distance(v,true);
                    if(d<dn){nearest=v;dn=d;}
                }
                if(prev==nearest)break;
            }
        }
        faiss::HNSW::MinimaxHeap candidates(std::max(k,ef));
        faiss::VisitedTable visited(index->ntotal);
        candidates.push(nearest,dn);res.add_result(dn,nearest);visited.set(nearest);
        struct Pending { int row,pivot; double threshold,sx,sz,rho,result_threshold,frontier_threshold; int ref,pred; };
        std::vector<Pending> pending;
        auto insert=[&](int row,float d) {
            if(d<res.threshold)res.add_result(d,row);
            candidates.push(row,d);
        };
        while(candidates.size()>0) {
            float dp;int pivot=candidates.pop_min(&dp);
            if(candidates.count_below(dp)>=ef)break;
            s[3]++;
            size_t begin,end;h.neighbor_range(pivot,0,&begin,&end);
            // Preserve the original batch-of-four grouping in no-op/shadow.
            std::vector<int> rows;
            for(size_t j=begin;j<end && h.neighbors[j]>=0;++j) {
                int x=h.neighbors[j];s[4]++;
                if(visited.get(x))continue;
                visited.set(x);rows.push_back(x);
            }
            for(size_t off=0;off<rows.size();off+=4) {
                int count=std::min(size_t(4),rows.size()-off);
                bool full=candidates.k==candidates.n && std::isfinite(res.threshold);
                double result_threshold=res.threshold,frontier_threshold=candidates.max();
                double threshold=full?std::max(result_threshold,frontier_threshold):INFINITY;
                int ref=-1;
                if(full)ref=candidates.max()>=res.threshold?candidates.ids[0]:int(res.heap_ids[0]);
                double sz=NAN;
                if(mode!=0 && mode!=3 && ref>=0)sz=project(ref);
                int accepted[4],na=0;double sx[4],rho[4];int predicted[4];
                for(int t=0;t<count;++t) {
                    int x=rows[off+t];sx[t]=rho[t]=NAN;predicted[t]=0;
                    bool safe_oracle=false;
                    if(mode==3 && full) {
                        if(!oracle || !std::isfinite(oracle[x]))throw std::runtime_error("oracle missing row");
                        safe_oracle=oracle[x]>=threshold;
                    }
                    if(mode!=0 && mode!=3 && full && ref>=0 && sz>0) {
                        sx[t]=project(x);s[5]++;
                        if(rule>=3) {
                            double xz=faiss::fvec_L2sqr(projected+size_t(x)*m,
                                                       projected+size_t(ref)*m,m);s[2]++;
                            rho[t]=(sx[t]+sz-xz)/(2*std::sqrt(sx[t]*sz));
                        }
                        if(std::isfinite(sx[t]) && sx[t]>0) {
                            if(rule==0)predicted[t]=sx[t]>cc*threshold;
                            if(rule==1)predicted[t]=sx[t]>fc*sz;
                            if(rule==2)predicted[t]=sx[t]>fc*threshold;
                            if(rule>=3 && std::isfinite(rho[t]) && std::abs(rho[t])<=1) {
                                double critical=rule==3?critical_beta(fc,rho[t]):
                                    1+(fc-1)*std::sqrt((1-rho[t])*(1+rho[t]));
                                predicted[t]=sx[t]>critical*sz;
                            }
                        }
                    }
                    if(predicted[t])s[6]++;
                    if(safe_oracle){s[7]++;continue;}
                    if(mode==2 && predicted[t]) {
                        pending.push_back({x,pivot,threshold,sx[t],sz,rho[t],result_threshold,frontier_threshold,ref,predicted[t]});
                        s[8]++;continue;
                    }
                    accepted[na++]=t;
                }
                float ds[4];
                if(na==4) {
                    dc->distances_batch_4(rows[off],rows[off+1],rows[off+2],rows[off+3],
                                          ds[0],ds[1],ds[2],ds[3]);s[0]+=4;
                } else for(int a=0;a<na;++a)ds[a]=distance(rows[off+accepted[a]]);
                for(int a=0;a<na;++a) {
                    int t=accepted[a],x=rows[off+t];float d=ds[a];
                    if(mode==3) {
                        if(!oracle || !std::isfinite(oracle[x]))throw std::runtime_error("oracle missing row");
                        if(d!=oracle[x])s[13]++;
                        // Omniscient counterfactual: charge the full kernel but
                        // retain baseline batch bits for exact path reproduction.
                        d=oracle[x];
                    }
                    int safe=full && d>=threshold;if(safe)s[9]++;
                    record(x,pivot,ref,d,threshold,sx[t],sz,rho[t],safe,predicted[t],0,result_threshold,frontier_threshold);
                    insert(x,d);
                }
            }
        }
        // A bounded terminal audit does not restore graph navigation and is not
        // claimed recall-preserving. Pending nodes remain explicitly accounted.
        std::sort(pending.begin(),pending.end(),[](const Pending&a,const Pending&b){
            return a.sx<b.sx || (a.sx==b.sx && a.row<b.row);});
        for(int j=0;j<std::min(audit,int(pending.size()));++j) {
            auto& p=pending[j];float d=distance(p.row);s[10]++;
            if(d<res.threshold)res.add_result(d,p.row);
            record(p.row,p.pivot,p.ref,d,p.threshold,p.sx,p.sz,p.rho,
                   d>=p.threshold,p.pred,1,p.result_threshold,p.frontier_threshold);
        }
        s[11]=pending.size();s[12]=used;res.end();
    }
};
}
extern "C" int route1_search(void* index,const float* query,const float* qp,
    const float* projected,int m,int k,int ef,int mode,int rule,double fc,double cc,
    int audit,const float* oracle,double* events,int capacity,float* D,int64_t* I,
    double* stats,char* error,int error_size) {
    try {
        if(!index || !query || k<=0 || ef<k || mode<0 || mode>4 || rule<0 || rule>4 ||
           (mode!=0 && mode!=3 && (!qp || !projected || m<=0)) || audit<0)
            throw std::runtime_error("invalid native arguments");
        Search s(static_cast<faiss::IndexHNSW*>(index),query,qp,projected,m,k,ef,mode,
                 rule,fc,cc,audit,oracle,events,capacity,stats);s.run(D,I);return 0;
    } catch(const std::exception& e) {
        if(error && error_size>0){std::snprintf(error,error_size,"%s",e.what());}return -1;
    }
}
