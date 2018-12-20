#!/bin/bash
# This script is for installing CGCNN source from Tian Xie at the Grossman research group at MIT.
# After installing CGCNN with this script, you should be able to use CGCNNFeaturizer

echo -e "\033[31mPlease make sure you are inside your desired venv (or conda env) before running this script!\033[0m"
read -p "Enter (full) path of install directory for CGCNN: " cgcnn_install_dir
cd "$cgcnn_install_dir"
while [[ $? -eq 1 ]]; do
    echo -e "\033[31m$cgcnn_install_dir not a valid installation path.\033[0m"
    read -p "Enter (full) path of install directory for CGCNN: " cgcnn_install_dir
    cd "$cgcnn_install_dir";
done
echo "Downloading CNCGG from source"
git clone https://github.com/txie-93/cgcnn      # conda install can be clunky to use
cd cgcnn

read -p "Add this install directory to your .bash_profile (recommended)? [y/n]: " add_pp_to_bp
if [[ "$add_pp_to_bp" == "y" ]]; then
    read -p "Enter bash_profile path (default ~/.bash_profile): " bppath
    if [[ -z "$bppath" ]]; then
        bppath="$HOME/.bash_profile"
    fi
    echo "# Adding CGCNN directory to PYTHONPATH for matminer" >> "$bppath"
    echo "export PYTHONPATH=\"\$PYTHONPATH:$(pwd)\"" >> "$bppath"
fi

read -p "Add this install directory to your ~/.bashrc (recommended)? [y/n]: " add_pp_to_brc
if [[ "$add_pp_to_brc" == "y" ]]; then
    read -p "Enter bash_profile path (default ~/.bashrc): " brcpath
    if [[ -z "$brcpath" ]]; then
        brcpath="$HOME/.bashrc"
    fi
    echo '# Adding CGCNN directory to PYTHONPATH for matminer' >> "$brcpath"
    echo "export PYTHONPATH=\"\$PYTHONPATH:$(pwd)\"" >> "$brcpath"
fi

pip install torch==0.3.1                        # Ensure CGCNN has the required version!
pip install torchvision
echo -e "\033[31mTo ensure CGCNN is found outside this shell, please ensure $(pwd) to your PYTHONPATH in your ~/.bash_profile and/or ~/.bashrc files!\033[0m"