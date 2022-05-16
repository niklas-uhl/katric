# IMPORT-DATA test output.txt

set terminal pdf size 28cm,18cm linewidth 2.0
set output "test.pdf"
set logscale x
set grid
set xlabel "threads"
set ylabel "time (s)"

## MULTIPLOT(partition, intersection_method, grainsize, partitioner) SELECT num_threads AS x, MEDIAN(time) as y, MULTIPLOT
## FROM (SELECT * FROM test WHERE input LIKE '%amazon%' and grainsize=1 and (partitioner LIKE 'affinity' OR partitioner LIKE 'auto' OR partitioner LIKE 'static'))
## GROUP BY MULTIPLOT, num_threads ORDER BY MULTIPLOT,num_threads; 
plot \
    'test-data.txt' index 0 title "partition=1D,intersection_method=binary_search,grainsize=1,partitioner=affinity" with linespoints, \
    'test-data.txt' index 1 title "partition=1D,intersection_method=binary_search,grainsize=1,partitioner=auto" with linespoints, \
    'test-data.txt' index 2 title "partition=1D,intersection_method=binary_search,grainsize=1,partitioner=static" with linespoints, \
    'test-data.txt' index 3 title "partition=1D,intersection_method=hybrid,grainsize=1,partitioner=affinity" with linespoints, \
    'test-data.txt' index 4 title "partition=1D,intersection_method=hybrid,grainsize=1,partitioner=auto" with linespoints, \
    'test-data.txt' index 5 title "partition=1D,intersection_method=hybrid,grainsize=1,partitioner=static" with linespoints, \
    'test-data.txt' index 6 title "partition=1D,intersection_method=merge,grainsize=1,partitioner=affinity" with linespoints, \
    'test-data.txt' index 7 title "partition=1D,intersection_method=merge,grainsize=1,partitioner=auto" with linespoints, \
    'test-data.txt' index 8 title "partition=1D,intersection_method=merge,grainsize=1,partitioner=static" with linespoints, \
    'test-data.txt' index 9 title "partition=2D,intersection_method=binary_search,grainsize=1,partitioner=affinity" with linespoints, \
    'test-data.txt' index 10 title "partition=2D,intersection_method=binary_search,grainsize=1,partitioner=auto" with linespoints, \
    'test-data.txt' index 11 title "partition=2D,intersection_method=binary_search,grainsize=1,partitioner=static" with linespoints, \
    'test-data.txt' index 12 title "partition=2D,intersection_method=hybrid,grainsize=1,partitioner=affinity" with linespoints, \
    'test-data.txt' index 13 title "partition=2D,intersection_method=hybrid,grainsize=1,partitioner=auto" with linespoints, \
    'test-data.txt' index 14 title "partition=2D,intersection_method=hybrid,grainsize=1,partitioner=static" with linespoints, \
    'test-data.txt' index 15 title "partition=2D,intersection_method=merge,grainsize=1,partitioner=affinity" with linespoints, \
    'test-data.txt' index 16 title "partition=2D,intersection_method=merge,grainsize=1,partitioner=auto" with linespoints, \
    'test-data.txt' index 17 title "partition=2D,intersection_method=merge,grainsize=1,partitioner=static" with linespoints
