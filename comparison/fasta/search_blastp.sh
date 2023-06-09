#!/bin/sh
for ifp in *.fasta
do
    echo $ifp
    ofp="${ifp%.*}".tab
    if [ ! -f $ofp ]
    then
        time blastp -query $ifp -db blastdb/nr -max_target_seqs 1 -outfmt 7 -out $ofp -num_threads 16
    fi
done
