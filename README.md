
# EzoGaming Music Isolator

This script uses [ZFTurbo's Music Source Separation Training repo](https://github.com/ZFTurbo/Music-Source-Separation-Training) and [wesleyr36's MDX23C similarity model](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/1#issuecomment-2417116936) to create isolations of music and automate the annoying part: pairing channels, exporting, inferring, re-pairing, re-shifting, exporting, dealing with audacity lag, and max speccing. It doesn't automate alignment (finding the sample offsets) but that may be a future idea.

The new version of the UVR5 GUI not working right was the final push for me to get around to installing the MSST repo and making this script.

It requires being in an environment for the MSST repo and also requires having it on your pc

Then you just supply the song path, sample offset to isolate, and "codename" for the isolation (to make telling apart easier when doing multiple) in the script file.

At the moment there are two scripts, run.py and run_batch.py, which do effectively the same thing with the key difference being that run_batch.py uses JSON files filled out with the offsets, codenames, song path, etc.

*Example json:*
```json
{
	"song_name": "fanfan_off",
	"song_path": "I:\\AllKemonoFriendsMusic\\Official Releases\\Fre! Fre! Best Friends (FLAC, MP3)\\A04 - Doubutsu Biscuits - Fun Fun Melody (off vocal ver.).flac",
	"isolations":
		[
			{"offset": 3586514, "codename": "VERSE-CHORUS"},
			{"offset": 87478, "codename": "4 BEATS"},
			{"offset": 43739, "codename": "2 BEATS"},
			{"offset": 21869, "codename": "1 BEATS"},
			{"offset": 174956, "codename": "8 BEATS"}
		]
}
```

Some future optimizations are going to have to be done such as:

- using a suitable module for the task of loading and editing the audio instead of ffmpegging into a numpy array that gets turned into a list and back (it's very slow!)
- cropping audio before inference to also save some time since the output will end up with a lot of silence.
	- basically, crop it from start of offset to end of original---just the overlap between the two files
- Library script and make the run and run batch less repeated
	- perhaps even integrate them into the same script
	- command-line support

In addition, some more features and QOL stuff to include...

- Sub-sample alignment where the audio is interpolated to shift in increments less than a sample
- Azimuth correction a la izotope RX? file B is adjusted dynamically to match the sample timing and/or gain of file A
- "Automatic" alignment based on BPM and beats, then adjusts on the sample level until difference between samples is lowest
- Script to run roformer and apollo on the output


# Installing MSST for noobs?

take this with a grain of salt - i have always been really bad when it comes to organizing my python installs and my brain is still stuck in that dark age where everything in the environment had to be *just right* or else everything would error out for one reason or another - like you would have to search issues on random github repos to find out a specific version of toiletlib was required because it deprecated something used by pycheeto

## anyways

when i installed the repo stuff i used a python 3.11.5 venv (i searched around my pc till i found a version of python that was 3.9 or later and venved it) then installed torch

`python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118                          `

not sure what torch version is exactly correct because that too likes to break with updates. also unsure about cuda version. cuda 11.8 seems to work fine on my RTX 3060 and ive just been using that every time i installed torch.

and installed the requirements.txt

then downgraded numpy to numpy==1.23.5 because, despite what i said earlier, i did have to change some modules because the version of librosa being used was using np.float. why didnt i upgrade librosa instead? dont know, maybe i was worried it would break other things, i havent seen numpy change ever other than deprecating things

i would suggest using conda but that installs everything to your C drive with no way to customize env locations, which is tough if you're strapped for storage (like me, who has 40% of a 500GB drive taken up by conda envs with random torch versions)