INFO
========================================================================


To login 
::

  ssh mhpcXX@hpc.c3e.cosint.it

where the XX stands for the number of your group, specified in the following table.

Each group will run on a dedicated node. 
To run on the computing node:
::

  qsub -I -l nodes=$NODE:ppn=24 -l walltime=8:0:0 -N $GROUP -q gpu

the available resources are assigned as follows:

+---------+----------+---------+----------------------+
|  GROUP  |  NODE    |	USER   |  MEMBERS             |
+=========+==========+=========+======================+
| group1  |  b11     |  mhpc01 | mowais, sparonuz     |
+---------+----------+---------+----------------------+
| group2  |  b12     |	mhpc02 | raversa, ndemo       | 
+---------+----------+---------+----------------------+
| group3  |  b13     |	mhpc03 | igirardi, mowais     |
+---------+----------+---------+----------------------+
| group4  |  b14     |	mhpc04 | jcarmona, aando      |
+---------+----------+---------+----------------------+
| group5  |  b15     |	mhpc05 | mbrenesn, pdicerbo   |
+---------+----------+---------+----------------------+ 
| group6  |  b17     |	mhpc06 | ansuini, plabus      |
+---------+----------+---------+----------------------+

