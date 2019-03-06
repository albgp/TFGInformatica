from fractions import gcd

def exists(n):
    p=1
    while p<n:
        p=p*2
    p//=2
    for z in range(0,p):
        if (gcd(z,p)!=1):
            continue
        exists=False
        for d in range(2*n):
            for r in range(1,n):
                if abs(z/p-d/r)<=1/(2*p):
                    exists=True
        if exists==False:
            return False
    return True


for n in range(2,1000):
    if not exists(n):
        print(n)
