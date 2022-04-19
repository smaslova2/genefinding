import numpy as np

def get_ppm(pfm):
    total = np.sum(pfm, axis=1)
    total = np.expand_dims(total, axis=1).repeat(4, axis=1)
    ppm = pfm/total

    return ppm

def get_pwm(pfm):
    ppm = get_ppm(pfm)
    pwm = np.log2(ppm/0.25)
    
    return ppm

def write_meme(ppm, output_file_path):
    meme_file = open(output_file_path + "filter_motifs_ppm.meme", 'w')
    meme_file.write("MEME version 4 \n")

    for i in range(0, 196):

        if np.sum(ppm[i,:,:]) > 0:
            meme_file.write("\n")
            meme_file.write("MOTIF filter%s \n" % i)
            meme_file.write("letter-probability matrix: alength= 4 w= %d \n" % np.count_nonzero(np.sum(ppm[i,:,:], axis=0)))

            for j in range(0, 22):
                if np.sum(ppm[i,:,j]) > 0:
                    meme_file.write(str(ppm[i,0,j]) + "\t" + str(ppm[i,1,j]) + "\t" + str(ppm[i,2,j]) + "\t" + str(ppm[i,3,j]) + "\n")
      
    meme_file.close()

