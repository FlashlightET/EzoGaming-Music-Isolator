import os
import subprocess as sp
import numpy as np

verbose=True

#path to "inference.py" in the music source separation training folder
mvsep_path=r'X:\Music-Source-Separation-Training\inference.py'

#path to desired song
song_path=r'I:\AllKemonoFriendsMusic\Official Releases\PPP In The Sky (FLAC)\05 - Gentoo Penguin - Hello！アイドル.flac'

#path to MDX23C Similarity by wesleyr36 weights
weight_path=r'L:\UVR\models\MDX_Net_Models\model_mdx23c_ep_271_l1_freq_72.2383.ckpt'

#path to MDX23C Similarity by wesleyr36 config
config_path=r'L:\UVR\models\MDX_Net_Models\model_data\mdx_c_configs\config_mdx23c_similarity.yaml'

#offset in 44k samples to isolate
offset=576032

#alignment codename for organization
codename="32 BEATS"


#make input output and final output directories if they dont exist
if not os.path.exists('inputs'):
    os.mkdir('inputs')
if not os.path.exists('outputs'):
    os.mkdir('outputs')
if not os.path.exists('final_outputs'):
    os.mkdir('final_outputs')


def clear_dir(directory):
    if verbose: print('> clearing '+directory+' dir...')
    for i in os.listdir(directory):
        if verbose: print('    - removing',i)
        os.remove(os.path.join(directory,i))

#clear input directory
clear_dir('inputs')
    
#clear output directory
clear_dir('outputs')
    
def audio_to_list(fp, verbose_indent=0):
    if verbose: print(' '*(verbose_indent*4)+'> converting file "',fp,'" to list...')
    cmd=[
            'ffmpeg', '-i', fp,
            '-f', 's16le',
            '-acodec', 'pcm_s16le',
            '-ac', '2',
            '-ar', '44100',
            '-'
        ]
    if verbose: print(' '*(verbose_indent*4)+'    - reading file...')
    process=sp.run(cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, check=True)
    if verbose: print(' '*(verbose_indent*4)+'    - converting...')
    raw_data=np.frombuffer(process.stdout, dtype=np.int16)
    inter_list=raw_data.tolist()
    samples=[[],[]]
    for i in range(0,len(inter_list),2):
        samples[0].append(inter_list[i])
        samples[1].append(inter_list[i+1])
        
    #return a list of [[L,L,L,...] [R,R,R,...]]
    return samples
    
    
def list_to_audio(samples, fp):
    if verbose: print('> converting list to file "',fp,'"...')
    #convert list back into an audio file (saves it)
    if verbose: print('    - interleaving...')
    interleaved=np.array(samples).T.flatten().astype(np.int16)
    raw_audio=interleaved.tobytes()
    if '.flac' in fp: #if saving as flac, use particular settings to optimize size.
        cmd=[
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 's16le',
                '-ar', '44100',
                '-ac', '2',
                '-i', '-',
                '-compression_level', '12', #we want a level 8 flac if we're doing flac
                fp
            ]
    else: #otherwise use encoder defaults
        cmd=[
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-f', 's16le',
                '-ar', '44100',
                '-ac', '2',
                '-i', '-',
                fp
            ]
    if verbose: print('    - converting...')
    process=sp.run(cmd,input=raw_audio,check=True)
    
    
def offset_song(song, samples):
    if verbose: print('> offsetting by',samples,'samples...')
    original_extended=[[0]*(len(song[0])+samples),[0]*(len(song[0])+samples)]
    offsetted_song=[[0]*(len(song[0])+samples),[0]*(len(song[0])+samples)]
    
    for i in range(len(song[0])):
        original_extended[0][i]=song[0][i]
        original_extended[1][i]=song[1][i]
        offsetted_song[0][i+samples]=song[0][i]
        offsetted_song[1][i+samples]=song[1][i]
    return original_extended, offsetted_song
    
    
def reverse_offset_song(song, samples):
    if verbose: print('> reverse offsetting by',samples,'samples...')
    offsetted_song=[[0]*(len(song[0])),[0]*(len(song[0]))]
    
    for i in range(samples,len(song[0])):
        offsetted_song[0][i-samples]=song[0][i]
        offsetted_song[1][i-samples]=song[1][i]
    return offsetted_song
    
#convert song to a format parsable by the script
song=audio_to_list(song_path)

#cache the length of the original song for later
input_length=len(song[0])

#make an offsetted version
extended, offsetted=offset_song(song,offset)

#make left and right pairs
L_pair=[extended[0],offsetted[0]]
R_pair=[extended[1],offsetted[1]]

list_to_audio(L_pair,os.path.join('inputs','left.wav'))
list_to_audio(R_pair,os.path.join('inputs','right.wav'))

if verbose: print('> running separation...')
if verbose: print('    - weight path:',weight_path)
if verbose: print('    - config path:',config_path)

os.system('python "'+mvsep_path+'" --model_type mdx23c --config_path "'+config_path+'" --start_check_point "'+weight_path+'" --input_folder "'+os.path.join(os.getcwd(),'inputs')+'" --store_dir "'+os.path.join(os.getcwd(),'outputs')+'"')

if verbose: print('> recombining outputs...')
if verbose: print('    - loading outputs...')
left_similarity_stereo=audio_to_list(os.path.join('outputs','left_similarity.wav'),1)
right_similarity_stereo=audio_to_list(os.path.join('outputs','right_similarity.wav'),1)

#average together the channels
if verbose: print('    - converting L to mono...')
left_similarity=[int((left_similarity_stereo[0][i]+left_similarity_stereo[1][i])/2) for i in range(len(left_similarity_stereo[0]))]
if verbose: print('    - converting R to mono...')
right_similarity=[int((right_similarity_stereo[0][i]+right_similarity_stereo[1][i])/2) for i in range(len(right_similarity_stereo[0]))]

#now we have 1D lists of LEFT and RIGHT similarity separately
#so now we merge them into a stereo similarity.
similarity=[left_similarity,right_similarity]

#next on the shopping list is applying this similarity to both regions of the audio, extending its range
#first de-offset the audio
deoffset_similarity=reverse_offset_song(similarity,offset)

#lets chop off the tails of the two similarity tracks now to potentially speed up processing
if verbose: print('> cropping...')
if verbose: print('    - similarity...')
similarity=[similarity[0][:input_length],similarity[1][:input_length]]
if verbose: print('    - deoffset similarity...')
deoffset_similarity=[deoffset_similarity[0][:input_length],deoffset_similarity[1][:input_length]]

#clear input folder again, we're gonna ensemble these
clear_dir('inputs')
list_to_audio(similarity,os.path.join('inputs','similarity.wav'))
list_to_audio(deoffset_similarity,os.path.join('inputs','deoffset_similarity.wav'))

#we are going to use max_fft (known as Max Spec in the UVR GUI)
files=[os.path.join('inputs','similarity.wav'),os.path.join('inputs','deoffset_similarity.wav')]
files=['"'+i+'"' for i in files]
if verbose: print('> ensembling...')
#note to self: numpy 1.26.3 was replaced with 1.23.5 to fix librosa error
os.system('python "'+mvsep_path.replace('inference','ensemble')+'" --files '+(' '.join(files))+' --weights 1 1 --type max_fft --output "'+os.path.join('outputs','ensemble.wav')+'"')


if verbose: print('> loading ensembled file...')
ensembled=audio_to_list(os.path.join('outputs','ensemble.wav'))
ensembled=[ensembled[0][:input_length],ensembled[1][:input_length]]

if verbose: print('> subtracting...')
#make a phase inversion of the song and the ensemble
subtracted=song
for i in range(len(song[0])):
    subtracted[0][i]=min(max(song[0][i]-ensembled[0][i],-32768),32767)
    subtracted[1][i]=min(max(song[1][i]-ensembled[1][i],-32768),32767)

#save it all to a permanent "final_outputs" directory
original_filename='.'.join(os.path.basename(song_path).split('.')[:-1])
if verbose: print('> saving final outputs...')
list_to_audio(ensembled,os.path.join('final_outputs',original_filename+' (Offset '+str(offset)+', Ensembled - '+codename+').flac'))
list_to_audio(subtracted,os.path.join('final_outputs',original_filename+' (Offset '+str(offset)+', Subtracted - '+codename+').flac'))


if verbose: print('> clearing temp directories...')
clear_dir('inputs')
clear_dir('outputs')

print('Done!')