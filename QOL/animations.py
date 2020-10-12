#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:23:22 2020

@author: Sevans
credits:
    create_mp4() is adapted from LMSAL_HUB vis_tools/movie_tools/cr_movie.py
"""

try:
    import cv2
    IMPORTED_CV2 = True
except:
    print("Failed to import cv2. QOL.animations functions will not work.", \
          "Other PythonQOL functions should continue to work as normal, though!")
    IMPORTED_CV2 = False
import os
import QOL.plots as pqol

#TODO: add overwrite parameter for saveframe. Also for pqol.savefig

def movie(movie=None, output=None, folder=None, overwrite=False, **kwargs):
    """"converts frames in folder/movie to movie at folder/output.
    
    **kwargs go to aqol.create_mp4
    
    Defaults (when parameter is passed as None)
    -------------------------------------------
    movie -> pqol.DEFAULT_SAVE_STR (== "Untitled" by default)
    output -> movie (so that frames_folder=='xY/' --> saves movie 'xY.mp4')
    folder -> pqol.savedir
    """
    movie  = movie  if movie  is not None else pqol.DEFAULT_SAVE_STR
    output = output if output is not None else movie
    folder = folder if folder is not None else pqol.savedir
   
    dir_path = os.path.join(folder,movie)
    print(folder, movie, dir_path, "output:",output)
    out_path = os.path.join(folder,output)
    create_mp4(dir_path = dir_path, output = out_path, **kwargs)
    print("movie saved to ", os.path.abspath(out_path))
    
makemovie = movie  #alternate name for movie function
mm = movie         #alternate name for movie function

def saveframe(movie=None, name=None, folder=None, **kwargs):
    """saves active plot as folder/movie/name.
    
    (actually converts name to next available name, imin=1.
    e.g. saveframe('a','b','c/') saves to c/b/1-a or c/b/2-a, etc)
    
    folder must end with '/'.
    if movie startswith '/', ignore folder.
    if name startswith '/', ignore movie and folder.
    
    Makes folder pqol.savedir if it doesn't already exist.
    
    **kwargs go to pqol.savefig (and then to plt.savefig)
    
    Defaults (when parameter is passed as None)
    -------------------------------------------
    movie -> pqol.DEFAULT_SAVE_STR (== "Untitled" by default)
    name -> movie (so that frames_folder=='xY/' --> frames are '1- xY', '2- xY', etc)
    folder -> pqol.savedir
    """
    imin = kwargs.pop('imin', 1)
    
    name = name if name is not None else ''
    if os.path.isabs(name):
        return pqol.savefig(name, imin=imin, **kwargs)
    else:
        movie = movie if movie is not None else pqol.DEFAULT_SAVE_STR
        if os.path.isabs(movie):
            folder, movie = os.path.split(movie)
        else:
            folder = folder if folder is not None else pqol.savedir
        if not os.path.isdir(folder): os.mkdir(folder)
        
        name = name if name != '' else movie 
        pqol.savefig(name, folder= os.path.join(folder, movie), imin=imin, **kwargs)

def soft_timing(Nframes, time, fpsmin=10, fpsmax=20):
    """determines time & fps; aims for target time, but forces fpsmin < fps < fpsmax.
    
    example usage: target 3 seconds, but force 10 < fps < 25:
        import QOL.animations as aqol
        for i in range(50):
            code_that_makes_plot_number_i()
            aqol.saveframe('moviename')
            plt.close()
        aqol.movie('moviename', **soft_timing(3, 10, 25))
        
    returns dict(time=time, fps=fps)
    """
    #determine timing or fps.
    if   time > Nframes/fpsmin: #makes sure the movie doesnt go too slow.
        (time, fps) = (None, fpsmin)
    elif time < Nframes/fpsmax: #makes sure the movie doesnt go too fast.
        (time, fps) = (None, fpsmax)
    else: #makes the movie 3 seconds long if it will have fpsmin < fps < fpsmax.
        (time, fps) = (time, 1) #fps will be ignored in aqol.movie since time is not None.    
    
def create_mp4(dir_path='.', output='output', ext='png',
               fps=10, time=None, visual=False, printfreq=500, verbose=True):
    """create movie from a folder filled with pngs named 1-*.ext, 2-*.ext,....
    
    example names: e.g. '1-  untitled.png', or '1-density.png'.
    
    Parameters
    ----------
    rootname
    dir_path : str. Default: '.' (points to current directory)
        path to directory (to use in os.listdir)
    output   : str. Default: 'output'
        name of output video.
        Only one '.' is allowed, and only to indicate file type.
        If no '.' in output name, '.mp4' is assumed.
    fps      : int. Default: 10
        frames per second. ignored if time is not None
        MUST NOT BE 2^N - THIS MAKES CRAZY GRAPHICS BUG. DON'T KNOW WHY.
        (Code now automatically ensures fps is not 2^N, via fps+=1 if needed.)
    time     : number, or None. Default: None
        length of full animation, in seconds. if None, use fps instead.
    visual   : bool. Default: False
        whether to show video
    printfreq: int. Default: 500
        update user on progress after every <printfreq> ms
        NOT YET IMPLEMENTED.
    verbose  : bool. Default: True
        whether to print extra info such as if any frames were reshaped.
    """
    
    output = output if '.' in output else output+'.mp4'
    #if output.endswith('.mp4'):
    #    codec='mp4v'

    #Get the files from directory
    files  = os.listdir(dir_path)
    imfiles = [f for f in files if f.split('-')[0].isnumeric()]
    imfiles.sort(key=lambda x: int(x.split('-')[0]))

    #Change framerate to fit the time in seconds if a time has been specified.
    #Overrides the -fps arg
    if time is not None:
        fps = int(len(imfiles) / int(time))
        if verbose: print("Adjusting framerate to " + str(fps))
        
    #Ensure framerate is 2^N for N>=1, which makes huge graphics bug for some reason.
    fr=2
    while fr<fps and fr<1e6: #1e6 max prevents infinite loop if fps is infinite.
        fr*=2 
    if fr==fps:
        fps += 1
        print("Framerate cannot be 2,4,8,16,... or makes weird graphics bug.\n"+\
              " >>> Adjusting fps by +1, to {:d}".format(fps))
              

    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, imfiles[0])
    frame = cv2.imread(image_path)

    if visual:
        cv2.imshow('video',frame)
    regular_size = os.path.getsize(image_path) #really not sure what this does.
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case # 0x00000020
    #fourcc = cv2.VideoWriter_fourcc(*'avc1') # Be sure to use lower case # 0x00000021
    #out = cv2.VideoWriter(output, fourcc, framerate, (width, height))
    #if codec=='avc1':
    if output.endswith('.mp4'):
    #if codec == 'mp4v':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #out = cv2.VideoWriter(output, 0x00000021, framerate, (width, height))
        out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    else:
        out = cv2.VideoWriter(output, 0x00000020, fps, (width, height))
    
    reshaped_frames=[]
    for n, imfile in enumerate(imfiles):
        #print("frame",n,end=' | ')
        image_path = os.path.join(dir_path, imfile)
        image_size = os.path.getsize(image_path)
        if image_size < regular_size / 1.5: #not sure what this does / why you'd want this
            print("Cancelled: " + imfile)
            continue

        frame = cv2.imread(image_path)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height)) #(w,h) is correct for cv2.resize.
            reshaped_frames+=[n]
        out.write(frame) # Write out frame to video
        if visual:
            cv2.imshow('video', frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                break

        
    if verbose and reshaped_frames != []:
        if len(reshaped_frames)<2*len(imfiles)/3:
            print("Resized frames (0-indexed):",reshaped_frames)
        else:
            original_shaped = [i for i in range(len(imfiles)) if not i in reshaped_frames]
            print("Resized all frames except (0-indexed):",original_shaped)

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()