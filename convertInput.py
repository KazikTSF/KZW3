def convert_input_to_data_files(input_file="data/input.txt", output_dir="data/", num_files=120):
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            content = file.readlines()
    except FileNotFoundError:
        raise Exception(f"File '{input_file}' not found")
    
    file_count = 0
    i = 0
    
    while i < len(content) and file_count < num_files:
        while i < len(content) and content[i].strip() == "":
            i += 1
        
        if i >= len(content):
            break
        line = content[i].strip()
        if line.startswith("data."):
            i += 1
        else:
            pass
        if i >= len(content):
            break
        
        header = content[i].strip()
        if not header:
            i += 1
            continue
        
        try:
            n, machines = map(int, header.split())
        except ValueError:
            raise Exception(f"Invalid header format at line {i+1}: '{header}'")
        
        # Read n task lines
        task_lines = []
        for j in range(n):
            if i + 1 + j >= len(content):
                raise Exception(f"Insufficient data: expected {n} tasks but reached end of file")
            
            task_line = content[i + 1 + j].strip()
            if not task_line:
                raise Exception(f"Empty task line at line {i + 2 + j}")
            
            task_lines.append(task_line)

        output_filename = f"{output_dir}data{file_count}.txt"
        with open(output_filename, "w", encoding="utf-8") as out_file:
            out_file.write(header + "\n")
            for task_line in task_lines:
                out_file.write(task_line + "\n")
        
        print(f"Created {output_filename}")
        
        file_count += 1
        i += n + 1

        while i < len(content) and not content[i].strip().startswith("data."):
            i += 1
    
    if file_count < num_files:
        print(f"Warning: Only created {file_count} files instead of {num_files}")
    else:
        print(f"Successfully created {file_count} files")

if __name__ == "__main__":
    convert_input_to_data_files()

