## Data formatting
In the **data_frame_making** change
> input_filename  =  "D:\IIT-D\Sem-4\JCD893-M.Tech Major Project Part-2\Anchor_Frame\Dataset\Demo/attack_dataset.csv"
> to the file where there is attack dataset

Run the script and it will generate .json file which will be used later for image generation


## Image Generation
Load the .json file which  has stored the formatted data
Run the script whih will generate images and the images will be stored in **attack** and **benign** folders respectively

## Creation of test data

The images which has been generated are stored in **attack** and **benign** folders.
Now make a new folder named **test** and shift the  **attack** and **benign** folders inside it.
 
##  Testing the IDS accuracy

Select one of the custom cnn model ending with a **.pth** extension and use it to run the test script.
