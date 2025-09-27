module gradient
  implicit none

contains
  function demsav(drdx,c,m,p,flag) result(grad)
    implicit none
    real*8,dimension(12,6)::drdx
    real*8,dimension(0:50)::c
    real*8,dimension(0:32)::m
    real*8,dimension(0:50)::p
    real*8::grad
    integer::flag
    ! ::::::::::::::::::::
    real*8,dimension(0:50)::dp

    call dbemsav(drdx,dp,m,p,flag)
    grad = dot_product(dp,c)

    return
  end function demsav

  subroutine dbemsav(drdx,dp,m,p,flag)
    implicit none
    real*8,dimension(12,6),intent(in)::drdx
    real*8,dimension(0:50),intent(out)::dp
    real*8,dimension(0:32),intent(in)::m
    real*8,dimension(0:50),intent(in)::p
    integer::flag
    ! ::::::::::::::::::::
    real*8,dimension(0:32)::dm

    call devmono(drdx,dm,m,flag)
    call devpoly(dm,p,dp)

    return
  end subroutine dbemsav

  subroutine devmono(drdx,dm,m,flag)
    implicit none
    real*8,dimension(12,6),intent(in)::drdx
    real*8,dimension(0:32),intent(out)::dm
    real*8,dimension(0:32),intent(in)::m
    integer::flag
    !::::::::::::::::::::
    real*8::a

    a = 1.0d0

    dm(0) = 0.0D0
    dm(1) = -m(1)/a*drdx(flag,6)
    dm(2) = -m(2)/a*drdx(flag,5)
    dm(3) = -m(3)/a*drdx(flag,3)
    dm(4) = -m(4)/a*drdx(flag,4)
    dm(5) = -m(5)/a*drdx(flag,2)
    dm(6) = -m(6)/a*drdx(flag,1)
    dm(7) = dm(1)*m(2) + m(1)*dm(2)
    dm(8) = dm(1)*m(3) + m(1)*dm(3)
    dm(9) = dm(2)*m(3) + m(2)*dm(3)
    dm(10) = dm(3)*m(4) + m(3)*dm(4)
    dm(11) = dm(2)*m(5) + m(2)*dm(5)
    dm(12) = dm(1)*m(6) + m(1)*dm(6)
    dm(13) = dm(4)*m(5) + m(4)*dm(5)
    dm(14) = dm(4)*m(6) + m(4)*dm(6)
    dm(15) = dm(5)*m(6) + m(5)*dm(6)
    dm(16) = dm(1)*m(9) + m(1)*dm(9)
    dm(17) = dm(1)*m(10) + m(1)*dm(10)
    dm(18) = dm(2)*m(10) + m(2)*dm(10)
    dm(19) = dm(1)*m(11) + m(1)*dm(11)
    dm(20) = dm(3)*m(11) + m(3)*dm(11)
    dm(21) = dm(2)*m(12) + m(2)*dm(12)
    dm(22) = dm(3)*m(12) + m(3)*dm(12)
    dm(23) = dm(2)*m(13) + m(2)*dm(13)
    dm(24) = dm(3)*m(13) + m(3)*dm(13)
    dm(25) = dm(1)*m(14) + m(1)*dm(14)
    dm(26) = dm(3)*m(14) + m(3)*dm(14)
    dm(27) = dm(1)*m(15) + m(1)*dm(15)
    dm(28) = dm(2)*m(15) + m(2)*dm(15)
    dm(29) = dm(4)*m(15) + m(4)*dm(15)
    dm(30) = dm(2)*m(24) + m(2)*dm(24)
    dm(31) = dm(1)*m(26) + m(1)*dm(26)
    dm(32) = dm(1)*m(28) + m(1)*dm(28)

    return
  end subroutine devmono

  subroutine devpoly(dm,p,dp)
    implicit none
    real*8,dimension(0:32),intent(in)::dm
    real*8,dimension(0:50),intent(in)::p
    real*8,dimension(0:50),intent(out)::dp
    !::::::::::::::::::::

    dp(0) = dm(0)
    dp(1) = dm(1) + dm(2) + dm(3)
    dp(2) = dm(4) + dm(5) + dm(6)
    dp(3) = dm(7) + dm(8) + dm(9)
    dp(4) = dm(10) + dm(11) + dm(12)
    dp(5) = dp(1)*p(2) + p(1)*dp(2) - dp(4)
    dp(6) = dm(13) + dm(14) + dm(15)
    dp(7) = dp(1)*p(1) + p(1)*dp(1) - dp(3) - dp(3)
    dp(8) = dp(2)*p(2) + p(2)*dp(2) - dp(6) - dp(6)
    dp(9) = dm(16)
    dp(10) = dm(17) + dm(18) + dm(19) + dm(20) + dm(21) + dm(22)
    dp(11) = dp(2)*p(3) + p(2)*dp(3) - dp(10)
    dp(12) = dm(23) + dm(24) + dm(25) + dm(26) + dm(27) + dm(28)
    dp(13) = dm(29)
    dp(14) = dp(1)*p(6) + p(1)*dp(6) - dp(12)
    dp(15) = dp(1)*p(3) + p(1)*dp(3) - dp(9) - dp(9) - dp(9)
    dp(16) = dp(1)*p(4) + p(1)*dp(4) - dp(10)
    dp(17) = dp(2)*p(7) + p(2)*dp(7) - dp(16)
    dp(18) = dp(2)*p(4) + p(2)*dp(4) - dp(12)
    dp(19) = dp(1)*p(8) + p(1)*dp(8) - dp(18)
    dp(20) = dp(2)*p(6) + p(2)*dp(6) - dp(13) - dp(13) - dp(13)
    dp(21) = dp(1)*p(7) + p(1)*dp(7) - dp(15)
    dp(22) = dp(2)*p(8) + p(2)*dp(8) - dp(20)
    dp(23) = dp(9)*p(2) + p(9)*dp(2)
    dp(24) = dm(30) + dm(31) + dm(32)
    dp(25) = dp(3)*p(6) + p(3)*dp(6) - dp(24)
    dp(26) = dp(13)*p(1) + p(13)*dp(1)
    dp(27) = dp(9)*p(1) + p(9)*dp(1)
    dp(28) = dp(3)*p(4) + p(3)*dp(4) - dp(23)
    dp(29) = dp(1)*p(10) + p(1)*dp(10) - dp(23) - dp(28) - dp(23)
    dp(30) = dp(1)*p(11) + p(1)*dp(11) - dp(23)
    dp(31) = dp(1)*p(12) + p(1)*dp(12) - dp(25) - dp(24) - dp(24)
    dp(32) = dp(1)*p(14) + p(1)*dp(14) - dp(25)
    dp(33) = dp(4)*p(5) + p(4)*dp(5) - dp(25) - dp(31)
    dp(34) = dp(2)*p(11) + p(2)*dp(11) - dp(25)
    dp(35) = dp(4)*p(6) + p(4)*dp(6) - dp(26)
    dp(36) = dp(2)*p(12) + p(2)*dp(12) - dp(26) - dp(35) - dp(26)
    dp(37) = dp(13)*p(2) + p(13)*dp(2)
    dp(38) = dp(2)*p(14) + p(2)*dp(14) - dp(26)
    dp(39) = dp(3)*p(3) + p(3)*dp(3) - dp(27) - dp(27)
    dp(40) = dp(3)*p(7) + p(3)*dp(7) - dp(27)
    dp(41) = dp(1)*p(16) + p(1)*dp(16) - dp(28)
    dp(42) = dp(2)*p(21) + p(2)*dp(21) - dp(41)
    dp(43) = dp(1)*p(18) + p(1)*dp(18) - dp(33)
    dp(44) = dp(7)*p(8) + p(7)*dp(8) - dp(43)
    dp(45) = dp(6)*p(6) + p(6)*dp(6) - dp(37) - dp(37)
    dp(46) = dp(2)*p(18) + p(2)*dp(18) - dp(35)
    dp(47) = dp(1)*p(22) + p(1)*dp(22) - dp(46)
    dp(48) = dp(6)*p(8) + p(6)*dp(8) - dp(37)
    dp(49) = dp(1)*p(21) + p(1)*dp(21) - dp(40)
    dp(50) = dp(2)*p(22) + p(2)*dp(22) - dp(48)

    return
  end subroutine devpoly

end module gradient
