import os
import shutil

def classify_in_or_ex(image_dir):
    ex_img = []
    in_img = []
    err_img = []

    for filename in os.listdir(image_dir):
        lower_name = filename.lower()
        # Fix error of name
        if "exteria" in lower_name:
            corrected_name = filename.replace("exteria", "exterior") # Right name
            src = os.path.join(image_dir, filename)
            dst = os.path.join(image_dir, corrected_name)
            os.rename(src, dst)
            print(f"Renamed: {filename} -> {corrected_name}")
            filename = corrected_name
            lower_name = filename.lower()

        if "exterior" in lower_name:
            ex_img.append(filename)
        elif "interior" in lower_name:
            in_img.append(filename)
        else:
            err_img.append(filename)
            print(f"ERROR: Cannot classify this image: {filename}")

    return ex_img, in_img, err_img

def copy_images(img_dir , file_list , output_dir) :
    """Copy images from list to output_dir
    """
    for file in file_list :
        src = os.path.join(image_dir, file)
        dst = os.path.join(output_dir, file)
        shutil.copyfile(src, dst)
        

if __name__ == '__main__' :
    # Input dir containing all original images
    image_dir = r"D:\kaggle_pottery\data\h690\sherd_images"
    
    # Output dir of two types
    ex_dir = r"D:\kaggle_pottery\data\dataset_classify\ex"
    in_dir = r"D:\kaggle_pottery\data\dataset_classify\in"
    
    ex_img , in_img , err_img = classify_in_or_ex(image_dir)
    
    print(f"Ex_DIR : {len(ex_img)} \n In_DIR : {len(in_img)} \n ERROR_DIR : {len(err_img)} \n")
    
    copy_images(image_dir , ex_img , ex_dir)
    copy_images(image_dir , in_img , in_dir)
     
    print("Over!")