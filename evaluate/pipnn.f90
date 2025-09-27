!***************************************************************************
!-->  program to get potential energy for a given geometry after NN fitting
!-->  global variables are declared in this module
!-->  Written by Kaisheng Song at 10 December 2022
!***************************************************************************
!
!--> NOTE: Check the variable defination section before using this program.
!    The most important variables: alpha,natom,npes,nmorse
!    Check the filename of weights and biases !!!
!
!***************************************************************************
module nnparam
implicit none
!***************************************************************************
!natom     ==>Number of atoms
!npes      ==>Number of PESs
!nmorse    ==>Number of morse-like potential
!***************************************************************************
real*8,parameter::alpha=1.d0,PI=3.141592653589793238d0,radian=PI/180.0d0,bohr=0.5291772d0
integer,parameter::natom=4,npes=1,nmorse=33
integer,parameter::nbond=natom*(natom-1)/2
integer ninput,noutput,nscale
integer nhid3,nlayer3,ifunc3
integer nwe3,nodemax3
integer,allocatable::nodes3a(:)
real*8,allocatable::weight3a(:,:,:,:),bias3a(:,:,:),pdel3a(:,:),pavg3a(:,:)
end module nnparam
!***************************************************************************
 subroutine evvdvdx(xcart,v)
!subroutine evvdvdx(xcart,v,forces_out,ndriv)
!***************************************************************************
!Subroutine to calculate the average potential energy v and analytical gradient dvdxa
!Call pes_init to read files and initialize before calling evvdvdx().
!v         ==>Average potential energy(in eV), v=sum(vpes)/npes
!vpes      ==>Potential energy(in eV) for each PES
!dvdxa     ==>Average gradient(in eV/ang), dvdxa=sum(dvdx)/npes
!dvdx      ==>Gradient(in eV/ang) for each PES
!ndriv     ==>ndriv=1 to compute the analytical gradient, otherwise ndriv=0
!***************************************************************************
use nnparam
implicit none
integer i,j,k
!integer,intent(in)::ndriv
real*8 dvdxa(1:natom*3)
real*8 vpes(npes),dvdx(1:natom*3,npes)
real*8,intent(in)::xcart(1:3,1:natom)
real*8,intent(out)::v
real*8 forces_out(3,natom)
real*8 xvec(3,nbond),xbond(1:nbond),rij(1:nbond)
real*8 m(0:nmorse-1),p(0:ninput)
real*8 expr(1:nbond)
real*8 txinput(1:ninput),dvdg(1,1:ninput,npes)
real*8 dpdx(1:ninput,1:natom*3)
! :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

xvec(1,1)=xcart(1,1)-xcart(1,2)
xvec(2,1)=xcart(2,1)-xcart(2,2)
xvec(3,1)=xcart(3,1)-xcart(3,2)
xvec(1,2)=xcart(1,1)-xcart(1,3)
xvec(2,2)=xcart(2,1)-xcart(2,3)
xvec(3,2)=xcart(3,1)-xcart(3,3)
xvec(1,3)=xcart(1,1)-xcart(1,4)
xvec(2,3)=xcart(2,1)-xcart(2,4)
xvec(3,3)=xcart(3,1)-xcart(3,4)
xvec(1,4)=xcart(1,2)-xcart(1,3)
xvec(2,4)=xcart(2,2)-xcart(2,3)
xvec(3,4)=xcart(3,2)-xcart(3,3)
xvec(1,5)=xcart(1,2)-xcart(1,4)
xvec(2,5)=xcart(2,2)-xcart(2,4)
xvec(3,5)=xcart(3,2)-xcart(3,4)
xvec(1,6)=xcart(1,3)-xcart(1,4)
xvec(2,6)=xcart(2,3)-xcart(2,4)
xvec(3,6)=xcart(3,3)-xcart(3,4)
rij(1) = dsqrt(xvec(1,1)*xvec(1,1)+xvec(2,1)*xvec(2,1)+xvec(3,1)*xvec(3,1))
rij(2) = dsqrt(xvec(1,2)*xvec(1,2)+xvec(2,2)*xvec(2,2)+xvec(3,2)*xvec(3,2))
rij(3) = dsqrt(xvec(1,3)*xvec(1,3)+xvec(2,3)*xvec(2,3)+xvec(3,3)*xvec(3,3))
rij(4) = dsqrt(xvec(1,4)*xvec(1,4)+xvec(2,4)*xvec(2,4)+xvec(3,4)*xvec(3,4))
rij(5) = dsqrt(xvec(1,5)*xvec(1,5)+xvec(2,5)*xvec(2,5)+xvec(3,5)*xvec(3,5))
rij(6) = dsqrt(xvec(1,6)*xvec(1,6)+xvec(2,6)*xvec(2,6)+xvec(3,6)*xvec(3,6))

xbond(:)=dexp(-rij(:)/alpha)
expr=-xbond
m=0.d0
p=0.d0
call bemsav(xbond,m,p)


txinput(1:ninput)=p(1:ninput)

!调用PES生成vpes,dedp
call pot3a(txinput,vpes)

v=dble(sum(vpes))/dble(npes) !几个势能面能量的平均值
!v = v-0.59262633d0
!if(ndriv.eq.1)then!反向传播求梯度
! !求dpdx 
! call devdpdx(xvec,rij,expr,m,p,alpha,dpdx)
! !求dvdx
! call evdvdx(dvdg,dpdx,dvdxa,dvdx)
!
!  do i = 1, natom
!     do j = 1, 3
!         forces_out(j, i) = dvdxa((i-1)*3 + j) ! X, Y, Z for atom i
!     end do
! end do
!endif

return
end subroutine evvdvdx
!****************************************************************!
!-->  read NN weights and biases from matlab output
subroutine pes_init
use nnparam
implicit none
integer i,j,ihid,iwe,inode1,inode2,ilay1,ilay2
integer ibasis,npd,iterm,ib,nfile1,nfile2
real*8,allocatable::tpw1(:,:),tpw2(:,:),nxw1(:,:),nxw2(:,:)
character f1*80

nfile1=4
nfile2=7
open(nfile1,file='weights.txt',status='old')
open(nfile2,file='biases.txt',status='old')

! read(nfile1,*)
! read(nfile2,*) 
 read(nfile1,*)ninput,nhid3,noutput
 nscale=ninput+noutput
 nlayer3=nhid3+2 !additional one for input layer and one for output 
 allocate(nodes3a(nlayer3),pdel3a(nscale,npes),pavg3a(nscale,npes))
 nodes3a(1)=ninput
 nodes3a(nlayer3)=noutput
 read(nfile1,*)(nodes3a(ihid),ihid=2,nhid3+1)
 nodemax3=0
 do i=1,nlayer3
  nodemax3=max(nodemax3,nodes3a(i))
 enddo
 allocate(weight3a(nodemax3,nodemax3,2:nlayer3,npes),bias3a(nodemax3,2:nlayer3,npes),tpw1(ninput,nodes3a(2)), &
          tpw2(nodes3a(2),nodes3a(3)),nxw1(nodes3a(2),ninput),nxw2(nodes3a(3),nodes3a(2)))
 weight3a=0.d0
 bias3a=0.d0
 read(nfile1,*)ifunc3,nwe3

do j=1,npes
 if(j.gt.1) then
  read(nfile1,*)
  read(nfile1,*)
  read(nfile1,*)
  read(nfile1,*)
  read(nfile2,*)
 endif
 read(nfile1,*)(pdel3a(i,j),i=1,nscale)
 read(nfile1,*)(pavg3a(i,j),i=1,nscale)
 iwe=0
 do ilay1=2,nlayer3
 ilay2=ilay1-1
 do inode1=1,nodes3a(ilay1)
 do inode2=1,nodes3a(ilay2) !
 read(nfile1,*)weight3a(inode2,inode1,ilay1,j)
 iwe=iwe+1
 enddo
 read(nfile2,*)bias3a(inode1,ilay1,j)
 iwe=iwe+1
 enddo
 enddo
 if (iwe.ne.nwe3) then
   write(*,*)'provided number of parameters ',nwe3
   write(*,*)'actual number of parameters ',iwe
   write(*,*)'nwe not equal to iwe, check input files or code'
   stop
 endif
 tpw1(:,:)=weight3a(1:ninput,1:nodes3a(2),2,j)
 tpw2(:,:)=weight3a(1:nodes3a(2),1:nodes3a(3),3,j)
 nxw1=transpose(tpw1)
 nxw2=transpose(tpw2)
 weight3a(1:nodes3a(2),1:ninput,2,j)=nxw1(:,:)
 weight3a(1:nodes3a(3),1:nodes3a(2),3,j)=nxw2(:,:)
enddo

close(nfile1)
close(nfile2)
!write(*,*)'initialization done'

end subroutine pes_init
!*************************************************************************
subroutine pot3a(x,vpot3)
!subroutine pot3a(x,vpot3,dvdg,ndriv)
use nnparam
implicit none
integer i,j,k,m,neu1,neu2,neu3,ndriv
real*8 x(ninput),vpot3(npes)
real*8 w1(1:nodes3a(2),1:ninput)
real*8 w2(1:nodes3a(3),1:nodes3a(2))
real*8 w3(1,1:nodes3a(3))
real*8 y0(1:ninput,1),y1t(1:nodes3a(2),1),y1(1:nodes3a(2),1)
real*8 y2t(1:nodes3a(3),1),y2(1:nodes3a(3),1)
real*8 df2(1:nodes3a(3)),diagdf2(1:nodes3a(3),1:nodes3a(3))
real*8 df1(1:nodes3a(2)),diagdf1(1:nodes3a(2),1:nodes3a(2))
real*8 y3,y3t(1,1)
real*8 dvdy2(1,1:nodes3a(3))
real*8 dy2dy1(1:nodes3a(3),1:nodes3a(2))
real*8 dvdy1(1,1:nodes3a(2))
real*8 dy1dy0(1:nodes3a(2),1:ninput),dvdy0(1,1:ninput)
real*8 dvdp(1:ninput,npes)
real*8 dvdg(1,1:ninput,npes)
real*8 ALPH,BETA 
real*8, external::tranfun

dvdg=0.d0
ALPH=1.D0
BETA=0.D0

!遍历m个势能面
do m=1,npes

  !将输入层归一化为y0
  y0(:,1)=(x(:)-pavg3a(1:ninput,m))/pdel3a(1:ninput,m)

  neu1=nodes3a(2);neu2=nodes3a(3);neu3=nodes3a(4)

  w1(:,:)=weight3a(1:neu1,1:ninput,2,m)
  w2(:,:)=weight3a(1:neu2,1:neu1,3,m)
  w3(1,:)=weight3a(1:neu2,1,4,m)

!-->.....evaluate the hidden layer
  call DGEMM('N','N',neu1,1,ninput,ALPH,w1,neu1,y0,ninput,BETA,y1t,neu1) 
  y1(:,1)=dtanh(y1t(:,1)+bias3a(1:neu1,2,m))      

  call DGEMM('N','N',neu2,1,neu1,ALPH,w2,neu2,y1,neu1,BETA,y2t,neu2) 
  y2(:,1)=dtanh(y2t(:,1)+bias3a(1:neu2,3,m)) 

  call DGEMM('N','N',1,1,neu2,ALPH,w3,1,y2,neu2,BETA,y3t,1)
  y3=y3t(1,1) 
  !求出势能面返回的能量，并逆转换为归一化前的原始数据
  vpot3(m)=(y3+bias3a(1,4,m))*pdel3a(nscale,m)+pavg3a(nscale,m)

  !if(ndriv.eq.1)then!反向传播求能量对输入层的偏导 dvdg

  !   df2=1-y2(:,1)*y2(:,1) !求dtanh的导数=1-dtanh**2
  !   diagdf2=0.d0
  !   forall(i=1:neu2) diagdf2(i,i)=df2(i) !向量转为对角矩阵

  !   !求出dvdy2 能量对第二层隐藏层的偏导 dy2dy1=w2*diagdf2
  !   call DGEMM('N','N',neu2,neu1,neu2,ALPH,diagdf2,neu2,w2,neu2,BETA,dy2dy1,neu2)

  !   df1=1-y1(:,1)*y1(:,1) !求dtanh的导数=1-dtanh**2
  !   diagdf1=0.d0

  !   forall(i=1:neu1) diagdf1(i,i)=df1(i) !向量转为对角矩阵

  !   !求第二层隐藏层对第一层隐藏层的偏导 dy1dy0=w1*diagdf1
  !   call DGEMM('N','N',neu1,ninput,neu1,ALPH,diagdf1,neu1,w1,neu1,BETA,dy1dy0,neu1)

  !   !求出能量对第一层隐藏层的偏导 dvdy1=dy3dy2*dy2dy1
  !   call DGEMM('N','N',1,neu1,neu2,ALPH,w3,1,dy2dy1,neu2,BETA,dvdy1,1)

  !   !求出能量对输入层的偏导dvdy0=dvdy1*dy1dy0  dy1dy0=w1为第一层隐藏层对输入层的偏导
  !   call DGEMM('N','N',1,ninput,neu1,ALPH,dvdy1,1,dy1dy0,neu1,BETA,dvdy0,1)

  !   !将dvdy0逆转换为归一化前的原始数据
  !   dvdg(1,:,m)=dvdy0(1,1:ninput)*pdel3a(nscale,m)/pdel3a(1:ninput,m)

  !endif
enddo      

return
end subroutine pot3a
!**************************************************************************
function tranfun(x,ifunc)
implicit none
integer ifunc
real*8 tranfun,x
!c    ifunc=1, transfer function is hyperbolic tangent function, 'tansig'
!c    ifunc=2, transfer function is log sigmoid function, 'logsig'
!c    ifunc=3, transfer function is pure linear function, 'purelin'. It is imposed to the output layer by default
if (ifunc.eq.1) then
tranfun=dtanh(x)
else if (ifunc.eq.2) then
tranfun=1d0/(1d0+exp(-x))
else if (ifunc.eq.3) then
tranfun=x
endif
return
end function tranfun
!**************************************************************************
FUNCTION LOGSIG(X)
REAL*8 X,LOGSIG
LOGSIG=1.d0/(1.d0+DEXP(-X))
RETURN
END FUNCTION LOGSIG
!**************************************************************************
!subroutine evdvdx(dvdg,dpdx,dvdxa,dvdx)
!use nnparam
!implicit none
!integer i,j,k
!real*8 ALPH,BETA 
!real*8 dvdx(1:natom*3,npes),xcart(1:3,1:natom)
!real*8 dEdx(1,1:natom*3),dpdr(1:ninput,1:nbond),dvdp(1,1:ninput)
!real*8 dpdx(1:ninput,1:natom*3)
!real*8 dvdxa(1:natom*3),dvdg(1,1:ninput,npes)
!
!ALPH=1.D0
!BETA=0.D0
!
!dvdxa=0.d0
!dEdx=0.d0
!dvdx=0.d0
!
!do k=1,npes !遍历k个势能面
!
!  dvdp(1,1:ninput)=dvdg(1,1:ninput,k)
!  !求出能量对坐标xyz的偏导数dEdx=dvdp*dpdx
!  call DGEMM('N','N',1,natom*3,ninput,ALPH,dvdp,1,dpdx,ninput,BETA,dEdx,1)
!
!  dvdx(:,k)=dEdx(1,:)!取负值
!
!enddo
!
!!求出几个势能面梯度的平均值
!forall(k=1:natom*3) dvdxa(k)=sum(dvdx(k,1:npes))/dble(npes) 
!
!return
!end subroutine evdvdx
!
