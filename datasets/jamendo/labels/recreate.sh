#!/bin/bash
here="${0%/*}"
wget -N http://www.mathieuramona.com/uploads/Main/jam_annote.tar.gz -P "$here"
tar -xzf "$here/"jam_annote.tar.gz -C "$here"
mv "$here/jamendo_lab/"*.lab "$here"
rm -r "$here"/{jam_annote.tar.gz,jamendo_lab}
