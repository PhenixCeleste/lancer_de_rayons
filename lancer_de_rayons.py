from math import *
import numpy as np
from PIL import Image
from os import chdir
from time import time
chdir('C:/Users/vivie/Documents/Vivien/lancer_rayon/')

def parallelepipede(C,u,v,w):
    """ Renvoie les 6 faces (parallelogrammes) d'un parallelepipede de point initial C, et de vecteurs d'orientation u, v et w. """
    L=[0]*6
    L[0]=(1,C,v,u)
    L[1]=(1,C,u,w)
    L[2]=(1,C,w,v)
    L[3]=(1,C+w,u,v)
    L[4]=(1,C+u,v,w)
    L[5]=(1,C+v,w,u)
    return L

# Quelques couleurs:
noir=np.array([0.,0.,0.])
blanc=np.array([1.,1.,1.])
jaune=np.array([1.,1.,0.])
rouge=np.array([1.,0.,0.])
bleu=np.array([0.,0.,1.])
cyan=np.array([0.,1.,1.])

""" Types d'éléments : 0-Sphere 1-Parallélogramme (orienté !) 2-Disque (orienté !) """

""" Définition de la scène. """
N=500 # Nombre pixels en hauteur et largeur de l'image (carrée)
Delta=500 # Taille de l'écran dans l'espace (largeur et hauteur)
Sol=-100. # Niveau du sol dans l'espace
omega=np.array([0.,10.,200.]) # Position de l'oeil
fond=np.array([0.05,0.15,0.05]) # Couleur de fond

# Définition des objets
S1=(0,np.array([-100.,75.+Sol,-200.]),75)
S2=(0,np.array([150.,25.+Sol,-40.]),25)
S3=(0,np.array([-25,40.+Sol,-50.]),40)
D1=(2,np.array([0.,0.+Sol,-280.]),np.array([0.,300.,0.]))
C1=parallelepipede(np.array([75.,0.+Sol,-100.]),np.array([50.*sqrt(3)/2,0.,50.*(-0.5)]),np.array([50.*(-0.5),0.,50.*(-sqrt(3)/2)]),np.array([0.,50.,0.]))

Objet=[S1,S2,S3,D1]+C1 # Liste des objets
KdObj=[np.array([0.4,0.2,0.8]),np.array([0.7,0.5,0.1]),np.array([0.8,0.1,0.3]),np.array([0.4,1.,1.])]+[np.array([1.,0.3,0.1])]*6 # Couleurs des objets

Source=[np.array([0.,200.,0.]),np.array([300.,100.,-300.])] # Sources de lumière
# ColSrc=[jaune,cyan]
ColSrc=[blanc,blanc] # Couleur des sources

# fonctions auxiliaures, manipulations de vecteurs
def vec(A,B):
    return B-A
def ps(v1,v2):
    return np.inner(v1,v2)
def pv(v1,v2):
    return np.cross(v1,v2)
def norme(v):
    return sqrt(ps(v,v))
def unitaire(v):
    return (1/norme(v))*v
def proj(v,u):
    return ps(u,v)*u
def pt(r,t):
    assert t>=0
    (S,u)=r
    return S+t*u
def dr(A,B):
    return unitaire(vec(A,B))
def ra(A,B):
    return A,dr(A,B)
def sp(A,B):
    return A,norme(vec(A,B))

def intersection(r,obj):
    """ Renvoie le point d'intersection entre le rayon r et l'objet obj, et la distance entre la source et l'objet. """
    typ=obj[0]
    (A,u)=r
    if typ==0:
        (typ,C,q)=obj
        CA=vec(C,A)
        a0=1
        b0=2*ps(u,CA)
        c0=norme(CA)**2-q**2
        delta=b0**2-4*a0*c0
        if delta<0:
            return None
        t1=(-b0+sqrt(delta))/2
        t2=(-b0-sqrt(delta))/2
        if t1<0 and t2<0:
            return None
        t=min(t1,t2)
        return (pt(r,t),t)
    elif typ==1:
        (typ,B,v,w)=obj
        n=pv(v,w)
        den=ps(n,u)
        AB=vec(A,B)
        if den==0:
            return None
        t=ps(n,AB)/den
        if t<0:
            return None
        BX=t*u-AB
        vp=proj(BX,unitaire(v))
        wp=proj(BX,unitaire(w))
        if norme(v)>=norme(vp) and norme(w)>=norme(wp) and ps(vp,v)>=0 and ps(wp,w)>=0:
            return (pt(r,t),t)
        else:
            return None
    elif typ==2:
        (typ,C,n)=obj
        den=ps(n,u)
        AC=vec(A,C)
        if den==0:
            return None
        t=ps(n,AC)/den
        if t<0:
            return None
        CX=t*u-AC
        if norme(n)>=norme(CX):
            return (pt(r,t),t)
        else:
            return None

def au_dessus(obj,P,src):
    """ Renvoie True si la source src est située au dessus du point P de l'objet obj, False sinon. """
    typ=obj[0]
    if typ==0:
        (typ,C,q)=obj
        CP=vec(C,P)
        Csrc=vec(C,src)
        CH=proj(Csrc,CP)
        return ps(CP,CH)>=0 and norme(CP)<=norme(CH)
    elif typ==1:
        (typ,B,v,w)=obj
        return ps(pv(v,w),vec(P,src))>=0
    elif typ==2:
        (typ,C,n)=obj
        return ps(n,vec(P,src))>=0

def visible(obj,j,P,src):
    """ Renvoie True si le point P de l'objet obj est éclairé par la source scr, False sinon. """
    if not au_dessus(obj[j],P,src):
        return False
    d=norme(vec(src,P))
    r=ra(src,P)
    for i in range(len(obj)):
        if i!=j:
            x=intersection(r,obj[i])
            if x!=None:
                y,t=x
                if t<=d:
                    return False
    return True

def couleur_diffusee(r,Cs,n,kd):
    (A,u)=r
    cosTheta=ps(n,-u)
    return np.maximum(0,cosTheta*(kd*Cs))

def rayon_reflechi(s,P,src):
    u=dr(src,P)
    (C,q)=s
    n=dr(C,P)
    cosTheta=ps(n,-u)
    w=u+2*cosTheta*n
    return P,w

def grille(i,j):
    d=Delta/N
    i0=N//2-i-0.5
    j0=j-N//2+0.5
    return np.array([j0*d,i0*d,0.])

def rayon_ecran(omega,i,j):
    return ra(omega,grille(i,j))

def interception(r):
    renvoi=None
    dmin=inf
    for i in range(len(Objet)):
        x=intersection(r,Objet[i])
        if x!=None:
            P,d=x
            if d<dmin:
                renvoi=(P,i)
                dmin=d
    return renvoi

def couleur_diffusion(P,j):
    Cd=np.copy(noir)
    obj=Objet[j]
    typ=obj[0]
    if typ==0:
        (typ,C,q)=obj
        n=dr(C,P)
    elif typ==1:
        (typ,B,v,w)=obj
        n=unitaire(pv(v,w))
    elif typ==2:
        (typ,C,v)=obj
        n=unitaire(v)
    for i in range(len(Source)):
        if visible(Objet,j,P,Source[i]):
            r=ra(Source[i],P)
            Cd=np.minimum(0.99,Cd+couleur_diffusee(r,ColSrc[i],n,KdObj[j]))
    return Cd

def lancer(omega,fond):
    im=np.zeros((N,N,3),dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            r=rayon_ecran(omega,i,j)
            x=interception(r)
            if x!=None:
                (P,k)=x
                coul=couleur_diffusion(P,k)
                im[i,j]=np.uint8(256*coul)
            else:
                im[i,j]=np.uint8(256*fond)
    return im

def affiche_scene():
    t=time()
    u=lancer(omega,fond)
    print(time()-t)
    im=Image.fromarray(u)
    im.show()

def sauve_scene(nom):
    t=time()
    u=lancer(omega,fond)
    print(time()-t)
    im=Image.fromarray(u)
    im.save(nom+'.png')
