#!/bin/bash
# In case you have the jamendo dataset already, just copy or symlink
# all the audio files here (without train/valid/test subdirectories)
here="${0%/*}"
for s in train valid test; do
	f=jam_"$s"_audio.tar.gz
	wget -N http://www.mathieuramona.com/uploads/Main/"$f" -P "$here"
	tar -xzf "$here"/"$f" -C "$here"
	mv "$here"/"$s"/*.* "$here"
	rm "$here"/"$f"
	rm -r "$here"/"$s"
done
chmod a-x "$here"/*.{ogg,mp3}
