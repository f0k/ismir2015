#!/bin/bash
here="${0%/*}"
for s in train valid test; do
	wget -N http://www.mathieuramona.com/uploads/Main/jam_"$s"_list.txt -P "$here"
	rm "$here"/$s 2>/dev/null
	while read -r fn; do
		fn=${fn%.trs}
		[ -f "$here"/../audio/"${fn}".ogg ] && fn="${fn}".ogg || fn="${fn}".mp3
		echo "$fn" >> "$here"/$s
	done < "$here"/jam_"$s"_list.txt
	rm "$here"/jam_"$s"_list.txt
done
