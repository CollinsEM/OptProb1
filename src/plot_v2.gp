set logscale x 2
set logscale y 2
plot 'data/v2_nt1.dat' using 1:5 with linespoints, 'data/v2_nt2.dat' using 1:5 with linespoints, 'data/v2_nt4.dat' using 1:5 with linespoints, 'data/v2_nt8.dat' using 1:5 with linespoints
