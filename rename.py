# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os


# Function to rename multiple files
def main():
    directory_name = "test/run_3_25_2020_odor_0_many_sine_f_perturbations/"
    for filename in os.listdir(directory_name):
        print(filename)
        dst = directory_name + "test_" +filename
        src = directory_name + filename

        # rename() function will
        # rename all the files
        os.rename(src, dst)


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()
