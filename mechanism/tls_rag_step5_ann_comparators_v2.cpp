/* Independent route1 v2 comparator experiment. FAISS v1.8.0 search control
 * flow is adapted under the MIT notice retained below; kernels/heaps come
 * from the unchanged installed FAISS. No future distances, labels or oracle
 * cache are accepted by this ABI. Individual pruning is empirical; only the
 * batch variant preserves original grouping for every evaluated batch.
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <faiss/IndexHNSW.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace {
using faiss::idx_t;
constexpr int EW=8;
struct Search {
    faiss::IndexHNSW* ix;
    const float *q,*qp,*xp,*norms;
    const double *table,*p;
    int m,k,ef,mode,rule,skip,stages,table_n;
    double *stats,*events;int capacity,used=0;
    std::unique_ptr<faiss::DistanceComputer> dc;
    struct Gate { bool valid=false;double sq=0,qnorm=0,constant=0,xcoefficient=0; } gate[3];
    Search(void* index,const float* query,const float* projected_query,
           const float* points,const float* metadata,const double* lookup,
           const int* opt,const double* params,double* s,double* ev,int cap)
        :ix(static_cast<faiss::IndexHNSW*>(index)),q(query),qp(projected_query),
         xp(points),norms(metadata),table(lookup),p(params),m(opt[0]),k(opt[1]),
         ef(opt[2]),mode(opt[3]),rule(opt[4]),skip(opt[5]),stages(opt[6]),
         table_n(opt[7]),stats(s),events(ev),capacity(cap),
         dc(ix->storage->get_distance_computer()) {
        std::fill(stats,stats+20,0.);dc->set_query(q);
        for(int j=0;j<stages;j++){gate[j].sq=p[8+4*j];gate[j].qnorm=std::sqrt(gate[j].sq);}
    }
    float full(int row,bool upper=false) {
        stats[0]++;if(upper)stats[1]++;return (*dc)(row);
    }
    void thresholds(double threshold) {
        // Conservative unit-correlation boundary for the actual float32
        // original coordinates. Auxiliary inputs are normalized BEFORE Pi.
        double t=threshold*(1+p[4])+p[5];
        double nx=std::clamp(p[0]-t,p[1],p[2]);
        double h=(p[0]+nx-t)/(2*std::sqrt(p[0]*nx));
        for(int j=0;j<stages;j++) {
            auto& g=gate[j];g.valid=false;
            if(!std::isfinite(h) || h<=-1 || h>=1 || !(g.sq>0))continue;
            int d=stages==1?m:(j==0?16:j==1?32:64);
            double fc=p[9+4*j],cc=p[10+4*j],z=p[11+4*j];
            if(rule==0) {g.constant=(g.sq-cc*(2-2*h))/2;g.xcoefficient=.5;}
            if(rule==1) {
                double c=fc*(1-h)/(1+h),theta=(1-c)/(2*(1+c));
                g.constant=theta*g.sq;g.xcoefficient=theta;
            }
            if(rule==2) {
                if(h<0)continue;
                g.constant=h*g.sq-z*std::sqrt((1-h*h)*g.sq/d);g.xcoefficient=0;
            }
            if(rule==3) {
                double delta=h*std::sqrt(d*g.sq/(1-h*h));
                double loc=std::floor((delta-p[6])/p[7]);
                if(!std::isfinite(loc) || loc<0 || loc>=table_n || delta>p[6]+p[7]*(table_n-1))continue;
                // Table entries already converted from nct to correlation.
                g.constant=table[j*table_n+int(loc)];g.xcoefficient=0;
            }
            if(rule==4) {
                // Ordinary Fisher correlation margin: asymptotic control,
                // deliberately not assigned an exact finite-sample guarantee.
                g.constant=std::tanh(std::atanh(h)-z/std::sqrt(d-3.));g.xcoefficient=0;
            }
            g.valid=true;
        }
    }
    bool inspect(int row,double& u,double& cut) {
        stats[6]++;double cumulative=0;int begin=0;
        for(int j=0;j<stages;j++) {
            const auto& g=gate[j];if(!g.valid){stats[11]++;return false;}
            int end=stages==1?m:(j==0?16:j==1?32:64);
            cumulative+=faiss::fvec_inner_product(qp+begin,xp+size_t(row)*m+begin,end-begin);
            stats[2]++;stats[3]+=end-begin;
            double scale=stages==1?1.:double(m)/end;
            u=cumulative*scale;begin=end;
            double norm=norms[size_t(row)*stages+j];stats[15]++;
            double product=g.qnorm*norm;
            if(!(norm>0) || !std::isfinite(norm) || !std::isfinite(u)){stats[11]++;return false;}
            cut=(rule>=3?g.constant*product:g.constant+g.xcoefficient*norm*norm)-p[3]*product;
            if(u<cut){stats[7]++;stats[12+j]++;return true;}
        }
        return false;
    }
    void record(int row,int pivot,double threshold,double u,double cut,bool pred,float d,bool rejected) {
        if(!events)return;
        if(used==capacity)throw std::runtime_error("event capacity exceeded");
        double values[EW]={double(row),double(pivot),threshold,u,cut,double(pred),d,double(rejected)};
        std::copy(values,values+EW,events+size_t(used++)*EW);
    }
    void run(float* D,idx_t* I) {
        auto& h=ix->hnsw;
        if(ix->metric_type!=faiss::METRIC_L2 || h.upper_beam!=1 || !h.search_bounded_queue ||
           !h.check_relative_distance || h.nb_neighbors(0)>64)
            throw std::runtime_error("unsupported original HNSW contract");
        using RH=faiss::HeapBlockResultHandler<faiss::HNSW::C>;
        RH handler(1,D,I,k);RH::SingleResultHandler res(handler);res.begin(0);
        int nearest=h.entry_point;if(nearest<0)throw std::runtime_error("empty index");
        float dn=full(nearest,true);
        for(int level=h.max_level;level>=1;level--)for(;;) {
            int prev=nearest;size_t b,e;h.neighbor_range(nearest,level,&b,&e);
            for(size_t j=b;j<e && h.neighbors[j]>=0;j++) {
                int row=h.neighbors[j];float d=full(row,true);if(d<dn){nearest=row;dn=d;}
            }
            if(prev==nearest)break;
        }
        faiss::HNSW::MinimaxHeap frontier(std::max(k,ef));faiss::VisitedTable visited(ix->ntotal);
        frontier.push(nearest,dn);res.add_result(dn,nearest);visited.set(nearest);
        while(frontier.size()>0) {
            float dp;int pivot=frontier.pop_min(&dp);if(frontier.count_below(dp)>=ef)break;
            stats[4]++;size_t b,e;h.neighbor_range(pivot,0,&b,&e);
            std::array<int,64> rows;int nr=0;
            for(size_t j=b;j<e && h.neighbors[j]>=0;j++) {
                int row=h.neighbors[j];stats[5]++;if(visited.get(row))continue;
                visited.set(row);rows[nr++]=row;
            }
            for(int off=0;off<nr;off+=4) {
                int count=std::min(4,nr-off),accepted[4],na=0;
                bool ready=frontier.k==frontier.n && std::isfinite(res.threshold);
                double threshold=ready?std::max(double(frontier.max()),double(res.threshold)):INFINITY;
                if(mode && ready)thresholds(threshold);
                bool pred[4]={};double us[4],cuts[4];int predictions=0;
                for(int j=0;j<count;j++) {
                    us[j]=cuts[j]=NAN;
                    if(mode && ready)pred[j]=inspect(rows[off+j],us[j],cuts[j]);
                    predictions+=pred[j];
                }
                bool batch_reject=mode==2 && skip==1 && predictions==count;
                if(batch_reject)stats[9]++;
                for(int j=0;j<count;j++) {
                    bool reject=mode==2 && (skip==0?pred[j]:batch_reject);
                    if(reject){stats[8]++;record(rows[off+j],pivot,threshold,us[j],cuts[j],pred[j],NAN,true);}
                    else accepted[na++]=j;
                }
                float ds[4];
                if(na==4) {
                    dc->distances_batch_4(rows[off],rows[off+1],rows[off+2],rows[off+3],ds[0],ds[1],ds[2],ds[3]);stats[0]+=4;
                } else {
                    if(na && na<count)stats[10]++;
                    for(int j=0;j<na;j++)ds[j]=full(rows[off+accepted[j]]);
                }
                for(int j=0;j<na;j++) {
                    int a=accepted[j],row=rows[off+a];float d=ds[j];
                    record(row,pivot,threshold,us[a],cuts[a],pred[a],d,false);
                    if(d<res.threshold)res.add_result(d,row);
                    frontier.push(row,d);
                }
            }
        }
        stats[16]=used;res.end();
    }
};
}
extern "C" int comparators_v2_search(void* ix,const float* q,const float* qp,const float* xp,
    const float* norms,const double* table,const int* opt,const double* params,float* D,int64_t* I,
    double* stats,double* events,int capacity,char* error,int error_size) {
    try {
        if(!ix || !q || !opt || !params || !D || !I || !stats || opt[1]<=0 || opt[2]<opt[1] ||
           opt[3]<0 || opt[3]>2 || opt[4]<0 || opt[4]>4 || opt[5]<0 || opt[5]>1 ||
           (opt[6]!=1 && opt[6]!=3) || (opt[6]==3 && opt[0]!=64) || capacity<0 ||
           (opt[3] && (!qp || !xp || !norms || !table || opt[0]<4)))
            throw std::runtime_error("invalid v2 arguments");
        Search s(ix,q,qp,xp,norms,table,opt,params,stats,events,capacity);s.run(D,I);return 0;
    } catch(const std::exception& e) {
        if(error && error_size>0)std::snprintf(error,error_size,"%s",e.what());return -1;
    }
}
