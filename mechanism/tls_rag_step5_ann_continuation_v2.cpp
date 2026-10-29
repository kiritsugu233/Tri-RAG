/* Approved route2 v2. Reuse the frozen v1 search implementation and MIT notice.
 * Extra endpoints observe/drain its pending list; full search order is unchanged.
 */
#include "tls_rag_step5_ann_continuation_v1.cpp"

extern "C" int continuation_v2_advance(void* state,uint64_t target,int boundary,
        double* stats,uint64_t* tokens,double* extra,char* error,int size) {
    try {
        if(!state || !stats || !tokens || !extra || boundary<0 || boundary>1)
            throw std::runtime_error("invalid v2 advance arguments");
        auto* s=static_cast<Session*>(state);s->advance(target);
        if(boundary)while(!s->done && s->off<s->pending_size)s->advance(s->ndis+1);
        s->status(stats,tokens);
        extra[0]=s->pending_size-s->off;extra[1]=s->pivot;
        extra[2]=s->off>=s->pending_size;return 0;
    } catch(const std::exception& e){fail(error,size,e.what());return -1;}
}

// Diagnostic only: full observed membership, without labels or future traversal.
extern "C" int continuation_v2_audit(void* state,int* pending,int* frontier,
        unsigned char* visited,char* error,int size) {
    try {
        if(!state || !pending || !frontier || !visited)
            throw std::runtime_error("invalid v2 diagnostic arguments");
        auto* s=static_cast<Session*>(state);
        std::fill(pending,pending+64,-1);
        std::copy(s->pending.begin()+s->off,s->pending.begin()+s->pending_size,pending);
        std::copy(s->frontier.ids.begin(),s->frontier.ids.end(),frontier);
        for(int j=0;j<s->ix->ntotal;j++)visited[j]=s->visited.get(j)?1:0;
        return 0;
    } catch(const std::exception& e){fail(error,size,e.what());return -1;}
}

extern "C" int continuation_v2_predict(const double* x,int nf,const int* roots,int nt,
        const int* feature,const double* threshold,const int* left,const int* right,
        const double* value,int nn,double initial,double rate,double* result) {
    if(!x || !roots || !feature || !threshold || !left || !right || !value || !result ||
       nf<1 || nt<1 || nn<1 || !std::isfinite(initial) || !std::isfinite(rate))return -1;
    double answer=initial;
    for(int t=0;t<nt;t++) {
        int j=roots[t],steps=0;
        while(true) {
            if(j<0 || j>=nn || ++steps>nn)return -2;
            if(left[j]<0){answer+=rate*value[j];break;}
            int f=feature[j];if(f<0 || f>=nf || !std::isfinite(x[f]))return -3;
            // sklearn's prediction path converts input features to float32.
            j=static_cast<float>(x[f])<=threshold[j]?left[j]:right[j];
        }
    }
    if(!std::isfinite(answer))return -4;*result=answer;return 0;
}
