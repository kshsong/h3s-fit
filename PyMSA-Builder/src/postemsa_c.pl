#!/usr/bin/perl -w

use strict;
use feature qw(say);

die " Syntax:\n   ./postemsa_c.pl [max degree] [molecule]\n" unless $#ARGV > 0;

my $degree = shift @ARGV;
my @atom = @ARGV;

my $stub = "MOL_" . join("_", @atom) . "_$degree";
my $mono = $stub . ".MONO";
my $poly = $stub . ".POLY";
my $hbas = "basis.h";
my $cbas = "basis.c";

# read in the MONO data file
my @dat_mono = ();

open(MONO, $mono) or die "Cannot open $mono to read!\n";

while (<MONO>) {
    chomp;
    my @mono = split;
    push(@dat_mono, \@mono);
}

close MONO;

my $nx = $#{$dat_mono[0]};
my $nm = $#dat_mono;

# read in the POLY data file
my @dat_poly = ();

open(POLY, $poly) or die "Cannot open $poly to read!\n";

while (<POLY>) {
    chomp;
    my @poly = split;
    push(@dat_poly, \@poly);
}

close POLY;

my $np = $#dat_poly;

# write header file `basis.h`

open(OUT, ">$hbas") or die "Cannot open $hbas to write!\n";

print_header_file(\*OUT, $nx, $nm, $np);

close OUT;

# write C file `basis.c`

open(OUT, ">$cbas") or die "Cannot open $cbas to write!\n";
print_definition(\*OUT, $nx, $nm, $np);

print_mono_head(\*OUT, $nx, $nm);

for (my $i = 0; $i < $nm + 1; $i++) {
    print_mono_body(\*OUT, $i, $dat_mono[$i]);
}

print_close(\*OUT);

print_poly_head(\*OUT, $nm, $nx);

for (my $i = 0; $i < $np + 1; $i++) {
    print_poly_body(\*OUT, $i, $dat_poly[$i]);
}

print_close(\*OUT);

# subroutines

sub print_header_file {
    my ($out, $nx, $nm, $np) = @_;

    print $out "#ifndef BASIS_H_\n";
    print $out "#define BASIS_H_\n";
    print $out "\n";
    print $out "#include <math.h>\n";
    print $out "\n";
    print $out "#define NX $nx\n";
    print $out "#define NM $nm\n";
    print $out "#define NP $np\n";
    print $out "\n";
    print $out "void evmono(double x[NX], double m[NM + 1]);\n";
    print $out "void evpoly(double m[NM + 1], double p[NP + 1]);\n";
    print $out "void bemsav(double x[NX], double p[NP + 1]); \n";
    print $out "\n";
    print $out "#endif /* BASIS_H_ */\n";
}

sub print_definition {
    my ($out, $nx, $nm, $np) = @_;

    print $out "#include \"basis.h\"\n";
    print $out "\n";
    print $out "double emsav(double x[NX], double c[NP + 1]) {\n";
    print $out "    double p[NP + 1];\n";
    print $out "    bemsav(x, p);\n";
    print $out "    double V = 0.0;\n";
    print $out "    for (int i = 0; i < NP + 1; i++) {\n";
    print $out "        V += c[i] * p[i];\n";
    print $out "    }\n";
    print $out "    return V;\n";
    print $out "}\n";
    print $out "\n";
    print $out "void bemsav(double x[NX], double p[NP + 1]) {\n";
    print $out "    double m[NM + 1];\n";
    print $out "    evmono(x, m);\n";
    print $out "    evpoly(m, p);\n";
    print $out "}\n";
    print $out "\n";
}

sub print_mono_head {
    my ($out, $nx, $nm) = @_;

    print $out "void evmono(double x[NX], double m[NM + 1]) {\n";
}

sub print_poly_head {
    my ($out, $nx, $nm) = @_;

    print $out "void evpoly(double m[NM + 1], double p[NP + 1]) {\n";
}

sub print_close {
    my $out = shift;
    print $out "}\n";
    print $out "\n";
}

sub print_mono_body {
    my ($out, $i_mono, $mono) = @_;

    print $out "    m[$i_mono] = ";

    my $stat = shift @$mono;

    if ($stat == 0) {
        my $new = 1;
        my $nz = 0;

        for(my $i = 0; $i < $#$mono + 1; $i++) {
            my $deg = $mono->[$i];
            
            next if $deg == 0;
            
            $nz++;
            
            if ($deg == 1) {
                if ($new) {
                    printf $out "x[%d]", $i;
                } else {
                    printf $out " * x[%d]", $i;
                }
                $new = 0;
            } else {
                if ($new) {
                    printf $out "pow(x[%d], $deg)", $i;
                } else {
                    printf $out " * pow(x[%d], $deg)", $i;
                }
                $new = 0;
            }
            # print $out " \\\n        " if $nz % 5 == 0 && $i < $#$mono;
        }
        print $out "1.0" if $new;
        print $out ";";
    } else {
        my $idx = $mono->[0];
        print $out "m[$idx]";
        for (my $i = 1; $i < $#$mono + 1; $i++) {
            my $idx = $mono->[$i];
            print $out " * m[$idx]";
            # print $out " \\\n        " if $i % 5 == 0 && $i < $#$mono;
        }
        print $out ";";
    }
    print $out "\n";
    unshift(@$mono, $stat);
}

sub print_poly_body {
    my ($out, $i_poly, $poly) = @_;

    my $stat = shift @$poly;
    printf $out "    p[$i_poly] = ";
    if ($stat == 2) {
        printf $out "m[%d]", $poly->[0];

        for (my $i = 1; $i <= $#$poly; $i++) {
            printf $out " + m[%d]", $poly->[$i];
            # print $out " \\\n        " if $i % 5 == 0 && $i < $#$poly;
        }
    } else {
        my $ima = shift @$poly;
        my $imb = shift @$poly;

        print $out "p[$ima] * p[$imb]";

        for (my $i = 0; $i <= $#$poly; $i++) {
            my $idx = $poly->[$i];
            print $out " - p[$idx]";
            # print $out " \\\n        " if $i % 5 == 0 && $i < $#$poly && $i > 0;
        }

        unshift(@$poly, $imb);
        unshift(@$poly, $ima);
    }
    print $out ";\n";
    unshift(@$poly, $stat);
}