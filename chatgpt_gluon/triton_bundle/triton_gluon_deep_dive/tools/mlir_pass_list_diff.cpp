
// build: g++ -O2 -std=c++17 -o mlir_pass_list_diff tools/mlir_pass_list_diff.cpp
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char** argv){
  if(argc!=3){ cerr<<"usage: "<<argv[0]<<" A.txt B.txt\n"; return 1; }
  auto load=[&](string p){ ifstream f(p); string s; map<string,int> c; while(getline(f,s)) if(!s.empty()) c[s]++; return c; };
  auto A=load(argv[1]), B=load(argv[2]);
  set<string> keys; for(auto&kv:A)keys.insert(kv.first); for(auto&kv:B)keys.insert(kv.first);
  for(auto&k:keys){ int a=A[k], b=B[k]; string tag=(a&&b)?"==":(a? "- ":"+ "); cout<<tag<<" "<<k<<"  "<<a<<" | "<<b<<"\n"; }
  return 0;
}
