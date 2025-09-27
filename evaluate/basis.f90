  subroutine bemsav(x,m,p)
    implicit none
    real*8,dimension(1:6),intent(in)::x
    real*8,dimension(0:50),intent(out)::p
    ! ::::::::::::::::::::
    real*8,dimension(0:32)::m

    call evmono(x,m)
    call evpoly(m,p)

    return
  end subroutine bemsav

  subroutine evmono(x,m)
    implicit none
    real*8,dimension(1:6),intent(in)::x
    real*8,dimension(0:32),intent(out)::m
    !::::::::::::::::::::

    m(0) = 1.0D0
    m(1) = x(6)
    m(2) = x(5)
    m(3) = x(3)
    m(4) = x(4)
    m(5) = x(2)
    m(6) = x(1)
    m(7) = m(1)*m(2)
    m(8) = m(1)*m(3)
    m(9) = m(2)*m(3)
    m(10) = m(3)*m(4)
    m(11) = m(2)*m(5)
    m(12) = m(1)*m(6)
    m(13) = m(4)*m(5)
    m(14) = m(4)*m(6)
    m(15) = m(5)*m(6)
    m(16) = m(1)*m(9)
    m(17) = m(1)*m(10)
    m(18) = m(2)*m(10)
    m(19) = m(1)*m(11)
    m(20) = m(3)*m(11)
    m(21) = m(2)*m(12)
    m(22) = m(3)*m(12)
    m(23) = m(2)*m(13)
    m(24) = m(3)*m(13)
    m(25) = m(1)*m(14)
    m(26) = m(3)*m(14)
    m(27) = m(1)*m(15)
    m(28) = m(2)*m(15)
    m(29) = m(4)*m(15)
    m(30) = m(2)*m(24)
    m(31) = m(1)*m(26)
    m(32) = m(1)*m(28)

    return
  end subroutine evmono

  subroutine evpoly(m,p)
    implicit none
    real*8,dimension(0:32),intent(in)::m
    real*8,dimension(0:50),intent(out)::p
    !::::::::::::::::::::

    p(0) = m(0)
    p(1) = m(1) + m(2) + m(3)
    p(2) = m(4) + m(5) + m(6)
    p(3) = m(7) + m(8) + m(9)
    p(4) = m(10) + m(11) + m(12)
    p(5) = p(1)*p(2) - p(4)
    p(6) = m(13) + m(14) + m(15)
    p(7) = p(1)*p(1) - p(3) - p(3)
    p(8) = p(2)*p(2) - p(6) - p(6)
    p(9) = m(16)
    p(10) = m(17) + m(18) + m(19) + m(20) + m(21) + m(22)
    p(11) = p(2)*p(3) - p(10)
    p(12) = m(23) + m(24) + m(25) + m(26) + m(27) + m(28)
    p(13) = m(29)
    p(14) = p(1)*p(6) - p(12)
    p(15) = p(1)*p(3) - p(9) - p(9) - p(9)
    p(16) = p(1)*p(4) - p(10)
    p(17) = p(2)*p(7) - p(16)
    p(18) = p(2)*p(4) - p(12)
    p(19) = p(1)*p(8) - p(18)
    p(20) = p(2)*p(6) - p(13) - p(13) - p(13)
    p(21) = p(1)*p(7) - p(15)
    p(22) = p(2)*p(8) - p(20)
    p(23) = p(9)*p(2)
    p(24) = m(30) + m(31) + m(32)
    p(25) = p(3)*p(6) - p(24)
    p(26) = p(13)*p(1)
    p(27) = p(9)*p(1)
    p(28) = p(3)*p(4) - p(23)
    p(29) = p(1)*p(10) - p(23) - p(28) - p(23)
    p(30) = p(1)*p(11) - p(23)
    p(31) = p(1)*p(12) - p(25) - p(24) - p(24)
    p(32) = p(1)*p(14) - p(25)
    p(33) = p(4)*p(5) - p(25) - p(31)
    p(34) = p(2)*p(11) - p(25)
    p(35) = p(4)*p(6) - p(26)
    p(36) = p(2)*p(12) - p(26) - p(35) - p(26)
    p(37) = p(13)*p(2)
    p(38) = p(2)*p(14) - p(26)
    p(39) = p(3)*p(3) - p(27) - p(27)
    p(40) = p(3)*p(7) - p(27)
    p(41) = p(1)*p(16) - p(28)
    p(42) = p(2)*p(21) - p(41)
    p(43) = p(1)*p(18) - p(33)
    p(44) = p(7)*p(8) - p(43)
    p(45) = p(6)*p(6) - p(37) - p(37)
    p(46) = p(2)*p(18) - p(35)
    p(47) = p(1)*p(22) - p(46)
    p(48) = p(6)*p(8) - p(37)
    p(49) = p(1)*p(21) - p(40)
    p(50) = p(2)*p(22) - p(48)

    return
  end subroutine evpoly

