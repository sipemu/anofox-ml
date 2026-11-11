#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int integer; typedef float real; typedef double doublereal;
extern doublereal slamch_(char*);
extern integer ilaenv_(integer*, char*, char*, integer*, integer*, integer*, integer*);
extern int ssygst_(integer*, char*, integer*, real*, integer*, real*, integer*, integer*);
extern int ssyev_(char*, char*, integer*, real*, integer*, real*, real*, integer*, integer*);
int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "consts")) {
    const char* q[] = {"E","S","B","P","N","R","M","U","L","O"};
    for (int i = 0; i < 10; ++i) { float v = slamch_((char*)q[i]); unsigned u; memcpy(&u,&v,4); printf("%s %.9g %08x\n", q[i], v, u); }
    integer one=1,two=2,three=3,m1=-1,n=200;
    printf("SSYGST nb %d\n", ilaenv_(&one,"SSYGST","L",&n,&m1,&m1,&m1));
    printf("SSYTRD nb %d nbmin %d nx %d\n", ilaenv_(&one,"SSYTRD","L",&n,&m1,&m1,&m1), ilaenv_(&two,"SSYTRD","L",&n,&m1,&m1,&m1), ilaenv_(&three,"SSYTRD","L",&n,&m1,&m1,&m1));
    printf("SORGQR nb %d nbmin %d nx %d\n", ilaenv_(&one,"SORGQR"," ",&n,&n,&n,&m1), ilaenv_(&two,"SORGQR"," ",&n,&n,&n,&m1), ilaenv_(&three,"SORGQR"," ",&n,&n,&n,&m1));
    return 0;
  }
  /* stdin: n, then n*n A bits, n*n B bits (hex u32). stdout: A bits, W bits, info */
  integer n; if (scanf("%d", &n) != 1) return 1;
  real* a = malloc(4*n*n); real* b = malloc(4*n*n); real* w = malloc(4*n);
  integer lwork = n*(n+2); real* work = malloc(4*lwork);
  for (int i = 0; i < n*n; ++i) { unsigned u; scanf("%x", &u); memcpy(&a[i],&u,4); }
  for (int i = 0; i < n*n; ++i) { unsigned u; scanf("%x", &u); memcpy(&b[i],&u,4); }
  integer itype = 1, info1, info2; char L = 'L', V = 'V';
  ssygst_(&itype, &L, &n, a, &n, b, &n, &info1);
  ssyev_(&V, &L, &n, a, &n, w, work, &lwork, &info2);
  for (int i = 0; i < n*n; ++i) { unsigned u; memcpy(&u,&a[i],4); printf("%08x ", u); } printf("\n");
  for (int i = 0; i < n; ++i) { unsigned u; memcpy(&u,&w[i],4); printf("%08x ", u); } printf("\n%d %d\n", info1, info2);
  return 0;
}
