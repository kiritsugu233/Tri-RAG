/* Independent, approved route2 fixed-beam continuation experiment.
 * Control flow adapted from FAISS v1.8.0; installed distance kernels unchanged.
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
#include <faiss/utils/Heap.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
using faiss::idx_t;
using Heap=faiss::HNSW::C;
struct Session {
    faiss::IndexHNSW* ix;int k,beam;bool debug,done=false;
    std::vector<float> q,dist;std::vector<idx_t> ids;
    std::unique_ptr<faiss::DistanceComputer> dc;
    faiss::HNSW::MinimaxHeap frontier;faiss::VisitedTable visited;
    std::array<int,64> pending{};int pending_size=0,off=0,pivot=-1,valid=0,reason=0;
    uint64_t ndis=0,upper=0,expanded=0,slots=0,updates=0,seen=0,advances=0,batches=0,scalars=0;
    uint64_t sequence=1469598103934665603ULL;double core_ms=0;

    static void hash_bytes(uint64_t& h,const void* value,size_t n) {
        const auto* b=static_cast<const unsigned char*>(value);
        for(size_t j=0;j<n;j++){h^=b[j];h*=1099511628211ULL;}
    }
    void counted(int row,float d) {
        if(debug){hash_bytes(sequence,&row,sizeof(row));hash_bytes(sequence,&d,sizeof(d));}
    }
    float full(int row,bool up=false) {
        float d=(*dc)(row);ndis++;scalars++;if(up)upper++;counted(row,d);return d;
    }
    void insert(int row,float d) {
        if(d<dist[0]) {
            if(ids[0]<0)valid++;
            faiss::heap_replace_top<Heap>(k,dist.data(),ids.data(),d,row);updates++;
        }
        frontier.push(row,d);
    }
    Session(void* index,const float* query,int outk,int ef,int dbg)
        :ix(static_cast<faiss::IndexHNSW*>(index)),k(outk),beam(ef),debug(dbg),
         q(query,query+ix->d),dist(k),ids(k),dc(ix->storage->get_distance_computer()),
         frontier(ef),visited(ix->ntotal) {
        auto t=Clock::now();auto& h=ix->hnsw;
        if(ix->metric_type!=faiss::METRIC_L2 || h.upper_beam!=1 || !h.search_bounded_queue ||
           !h.check_relative_distance || h.nb_neighbors(0)>64 || ix->ntotal<k)
            throw std::runtime_error("unsupported fixed HNSW contract");
        for(float x:q)if(!std::isfinite(x))throw std::runtime_error("invalid query");
        faiss::heap_heapify<Heap>(k,dist.data(),ids.data());dc->set_query(q.data());
        int nearest=h.entry_point;if(nearest<0)throw std::runtime_error("empty index");
        float dn=full(nearest,true);
        for(int level=h.max_level;level>=1;level--)for(;;) {
            int prev=nearest;size_t b,e;h.neighbor_range(nearest,level,&b,&e);
            for(size_t j=b;j<e && h.neighbors[j]>=0;j++) {
                int row=h.neighbors[j];float d=full(row,true);
                if(d<dn){nearest=row;dn=d;}
            }
            if(nearest==prev)break;
        }
        insert(nearest,dn);visited.set(nearest);seen++;
        core_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t).count();
    }
    void advance(uint64_t target) {
        auto t=Clock::now();advances++;
        while(!done) {
            if(target && ndis>=target && valid==k)break;
            if(off>=pending_size) {
                off=pending_size=0;
                if(!frontier.size()){done=true;reason=1;break;}
                float dp;pivot=frontier.pop_min(&dp);
                if(frontier.count_below(dp)>=beam){done=true;reason=2;break;}
                expanded++;size_t b,e;ix->hnsw.neighbor_range(pivot,0,&b,&e);
                for(size_t j=b;j<e && ix->hnsw.neighbors[j]>=0;j++) {
                    int row=ix->hnsw.neighbors[j];slots++;
                    if(visited.get(row))continue;
                    visited.set(row);seen++;pending[pending_size++]=row;
                }
                if(!pending_size)continue;
            }
            int count=std::min(4,pending_size-off);float ds[4];
            if(count==4) {
                dc->distances_batch_4(pending[off],pending[off+1],pending[off+2],pending[off+3],
                    ds[0],ds[1],ds[2],ds[3]);ndis+=4;batches++;
                for(int j=0;j<4;j++)counted(pending[off+j],ds[j]);
            } else for(int j=0;j<count;j++)ds[j]=full(pending[off+j]);
            for(int j=0;j<count;j++)insert(pending[off+j],ds[j]);off+=count;
        }
        core_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t).count();
    }
    void status(double* s,uint64_t* tokens) const {
        float front=INFINITY;
        for(int j=0;j<frontier.k;j++)if(frontier.ids[j]>=0)front=std::min(front,frontier.dis[j]);
        double v[]={double(ndis),double(upper),double(expanded),double(slots),double(updates),
            double(seen),double(advances),core_ms,double(batches),double(scalars),double(done),
            double(reason),double(valid),double(frontier.size()),double(dist[0]),double(front)};
        std::copy(v,v+16,s);tokens[0]=debug?sequence:0;tokens[1]=0;
        if(debug) {
            uint64_t h=1469598103934665603ULL;
            hash_bytes(h,dist.data(),dist.size()*sizeof(float));hash_bytes(h,ids.data(),ids.size()*sizeof(idx_t));
            hash_bytes(h,frontier.ids.data(),frontier.ids.size()*sizeof(int));
            hash_bytes(h,frontier.dis.data(),frontier.dis.size()*sizeof(float));
            hash_bytes(h,visited.visited.data(),visited.visited.size()*sizeof(visited.visited[0]));
            hash_bytes(h,pending.data(),pending_size*sizeof(int));hash_bytes(h,&off,sizeof(off));
            hash_bytes(h,&done,sizeof(done));tokens[1]=h;
        }
    }
    void snapshot(float* D,idx_t* I) const {
        if(valid<k)throw std::runtime_error("result pool not ready");
        std::copy(dist.begin(),dist.end(),D);std::copy(ids.begin(),ids.end(),I);
        faiss::heap_reorder<Heap>(k,D,I); // Only copies are reordered.
    }
};
void fail(char* error,int size,const char* message) {
    if(error && size>0)std::snprintf(error,size,"%s",message);
}
}
extern "C" void* continuation_create(void* index,const float* q,int k,int beam,int debug,char* error,int size) {
    try {
        if(!index || !q || k<1 || beam<k || debug<0 || debug>1)throw std::runtime_error("invalid session arguments");
        return new Session(index,q,k,beam,debug);
    } catch(const std::exception& e){fail(error,size,e.what());return nullptr;}
}
extern "C" int continuation_advance(void* state,uint64_t target,double* stats,uint64_t* tokens,char* error,int size) {
    try {
        if(!state || !stats || !tokens)throw std::runtime_error("invalid advance arguments");
        auto* s=static_cast<Session*>(state);s->advance(target);s->status(stats,tokens);return 0;
    } catch(const std::exception& e){fail(error,size,e.what());return -1;}
}
extern "C" int continuation_snapshot(void* state,float* D,idx_t* I,double* stats,uint64_t* tokens,char* error,int size) {
    try {
        if(!state || !D || !I || !stats || !tokens)throw std::runtime_error("invalid snapshot arguments");
        auto* s=static_cast<Session*>(state);s->snapshot(D,I);s->status(stats,tokens);return 0;
    } catch(const std::exception& e){fail(error,size,e.what());return -1;}
}
extern "C" void continuation_destroy(void* state){delete static_cast<Session*>(state);}
