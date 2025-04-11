import yaml

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
    
input_path = config["txt_input_path"]
output_path = config["txt_parsed_path"]

with open(input_path, "r") as fin, open(output_path, "w") as fout:
    fout.write("IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    for line in fin:
        tokens = line.strip().split()
        if line.strip() and not line.strip().startswith("#") and len(tokens) == 10:
            fout.write(line)
