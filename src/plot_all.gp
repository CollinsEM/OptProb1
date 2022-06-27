set logscale x 2
set logscale y 2
plot 'v1.dat' using 1:6 with linespoints, 'v2_nt1.dat' using 1:6 with linespoints, 'v2_nt2.dat' using 1:6 with linespoints, 'v2_nt4.dat' using 1:6 with linespoints, 'v2_nt8.dat' using 1:6 with linespoints, 'v3.dat' using 1:6 with linespoints
