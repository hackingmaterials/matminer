import os
import glob

"""
Script used to transform the data from the original repository to 
an easier-to-use database for matminer. Very ugly but should work, 
at least with the commit 5a22ce0a3fa41fe6e9d26330686b32d83388e760 of 
https://github.com/polyanskiy/refractiveindex.info-database

Place this script in your refractiveindex repo, and run it.
A folder database_matminer should appear containing the new database.

We use only files from main and other/semiconductors alloys, other/alloys and other/intermetallics.
Within these data, some files are not used because they are outliers when looking at
plots of properties vs composition. 
Some files are renamed to represent the composition in an cleaner way.

Overall, this is very ugly and represents manual manipulations. 
This is just added so that anyone can update the database used by matminer 
with new data added on the refractiveindex repository. 
Add only data that you are sure is representative of 3D crystals, there are no automatic checks!
"""

os.system("cp -r database database_copy")
os.mkdir("database_matminer")

# Get the list of files to remove
files_to_remove = [
    "data/main/Al2O3/Querry.yml",
    "data/main/Au/*-*nm*.yml",
    "data/main/C/Arakawa.yml",
    "data/main/C/Djurisic-e.yml",
    "data/main/C/Djurisic-o.yml",
    "data/main/C/El-Sayed.yml",
    "data/main/C/Ermolaev.yml",
    "data/main/C/Hagemann.yml",
    "data/main/C/Querry*.yml",
    "data/main/C/Song*.yml",
    "data/main/C/Weber.yml",
    "data/main/Ge/*nm*.yml",
    "data/main/InSb/Adachi.yml",
    "data/main/MoS2/Beal.yml",
    "data/main/MoS2/Ermolaev*.yml",
    "data/main/MoS2/Hsu*.yml",
    "data/main/MoS2/Islam*.yml",
    "data/main/MoS2/Jung.yml",
    "data/main/MoS2/Song-*L.yml",
    "data/main/MoS2/Yim*.yml",
    "data/main/MoS2/Zhang.yml",
    "data/main/Na/Inagaki-liquid.yml",
    "data/main/Bi2Se3/Fang-2L.yml",
    "data/main/Pb/Mathewson-140K.yml",
    "data/main/Si/Pierce.yml",
    "data/main/WSe2/*L*.yml",
    "data/main/*/*Werner*.yml",
    "data/other/alloys/Ni-Fe/*10nm*",
    "data/other/alloys/Ni-Fe/*gold150nm*",
    "data/other/intermetallics/AuAl2/Supansomboon.yml",
    "data/main/*/Munkhbat-e.yml",
    "data/main/*/Munkhbat-gamma.yml"
]

flat_list = []
for f in files_to_remove:
    flat_list += glob.glob(os.path.join("database_copy", f))

flat_list += ["\'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Aspnes-0.yml\'"]
flat_list += ["\'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Papatryfonos-0.yml\'"]
flat_list.remove("database_copy/data/main/Ta/Werner.yml")
flat_list.remove("database_copy/data/main/Ta/Werner-DFT.yml")

files_to_remove = flat_list

for f in files_to_remove:
    os.system("rm " + f)

# Also change a file
os.system("sed -i \'s/760/780/g\' database_copy/data/main/C/Peter.yml")

# Move some files
os.system("mv database_copy/data/other/alloys/Au-Ag/Rioux-Au0Ag100.yml database_copy/data/main/Ag/Rioux.yml")
os.system("mv database_copy/data/other/alloys/Au-Ag/Rioux-Au100Ag0.yml database_copy/data/main/Au/Rioux.yml")
os.system("mv \'database_copy/data/other/semiconductor alloys/AlSb-GaSb/Ferrini-0.yml\' database_copy/data/main/GaSb/Ferrini.yml")

# Remove entire folders
folders_to_remove = [
    "data/other/alloys/Pd-H",
    "data/other/alloys/V-H",
    "data/other/alloys/Zr-H",
    "data/other/alloys/Nb-Sn",
    "data/other/alloys/V-Ga"
]

for f in folders_to_remove:
    os.system("rm -rf database_copy/" + f)

# Now the database has been filtered.
# We move the files into database_matminer,
# with some renaming...
os.system("cp -r database_copy/data/main/* database_matminer/")
os.system("cp -r database_copy/data/other/intermetallics/* database_matminer/")
os.mkdir("database_matminer/Au2Ag3")
os.mkdir("database_matminer/Au3Ag2")
os.mkdir("database_matminer/Au3Ag7")
os.mkdir("database_matminer/Au4Ag")
os.mkdir("database_matminer/Au7Ag3")
os.mkdir("database_matminer/Au9Ag")
os.mkdir("database_matminer/AuAg")
os.mkdir("database_matminer/AuAg4")
os.mkdir("database_matminer/AuAg9")
os.system("cp database_copy/data/other/alloys/Au-Ag/Rioux-Au40Ag60.yml database_matminer/Au2Ag3/")
os.system("cp database_copy/data/other/alloys/Au-Ag/Rioux-Au60Ag40.yml database_matminer/Au3Ag2/")
os.system("cp database_copy/data/other/alloys/Au-Ag/Rioux-Au30Ag70.yml database_matminer/Au3Ag7/")
os.system("cp database_copy/data/other/alloys/Au-Ag/Rioux-Au80Ag20.yml database_matminer/Au4Ag/")
os.system("cp database_copy/data/other/alloys/Au-Ag/Rioux-Au70Ag30.yml database_matminer/Au7Ag3/")
os.system("cp database_copy/data/other/alloys/Au-Ag/Rioux-Au90Ag10.yml database_matminer/Au9Ag/")
os.system("cp database_copy/data/other/alloys/Au-Ag/Rioux-Au50Ag50.yml database_matminer/AuAg/")
os.system("cp database_copy/data/other/alloys/Au-Ag/Rioux-Au20Ag80.yml database_matminer/AuAg4/")
os.system("cp database_copy/data/other/alloys/Au-Ag/Rioux-Au10Ag90.yml database_matminer/AuAg9/")

os.mkdir("database_matminer/Cu17Zn3")
os.mkdir("database_matminer/Cu7Zn3")
os.mkdir("database_matminer/Cu9Zn")
os.system("cp database_copy/data/other/alloys/Cu-Zn/Querry-Cu85Zn15.yml database_matminer/Cu17Zn3/")
os.system("cp database_copy/data/other/alloys/Cu-Zn/Querry-Cu70Zn30.yml database_matminer/Cu7Zn3/")
os.system("cp database_copy/data/other/alloys/Cu-Zn/Querry-Cu90Zn10.yml database_matminer/Cu9Zn/")

os.mkdir("database_matminer/Ni4Fe")
os.system("cp database_copy/data/other/alloys/Ni-Fe/Tikuisis_bare150nm.yml database_matminer/Ni4Fe/")


os.mkdir("database_matminer/Al198Ga802As1000")
os.mkdir("database_matminer/Al219Ga781As1000")
os.mkdir("database_matminer/Al315Ga685As1000")
os.mkdir("database_matminer/Al342Ga658As1000")
os.mkdir("database_matminer/Al411Ga589As1000")
os.mkdir("database_matminer/Al419Ga581As1000")
os.mkdir("database_matminer/Al452Ga548As1000")
os.mkdir("database_matminer/Al491Ga509As1000")
os.mkdir("database_matminer/Al590Ga410As1000")
os.mkdir("database_matminer/Al700Ga300As1000")
os.mkdir("database_matminer/Al804Ga196As1000")
os.mkdir("database_matminer/Al97Ga903As1000")
os.mkdir("database_matminer/Al99Ga901As1000")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Aspnes-19.8.yml\' database_matminer/Al198Ga802As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Papatryfonos-21.9.yml\' database_matminer/Al219Ga781As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Adachi-0.315.yml\' database_matminer/Al315Ga685As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Aspnes-31.5.yml\' database_matminer/Al315Ga685As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Papatryfonos-34.2.yml\' database_matminer/Al342Ga658As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Papatryfonos-41.1.yml\' database_matminer/Al411Ga589As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Aspnes-41.9.yml\' database_matminer/Al419Ga581As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Papatryfonos-45.2.yml\' database_matminer/Al452Ga548As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Aspnes-49.1.yml\' database_matminer/Al491Ga509As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Aspnes-59.0.yml\' database_matminer/Al590Ga410As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Adachi-0.700.yml\' database_matminer/Al700Ga300As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Aspnes-70.0.yml\' database_matminer/Al700Ga300As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Aspnes-80.4.yml\' database_matminer/Al804Ga196As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Papatryfonos-9.7.yml\' database_matminer/Al97Ga903As1000/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlAs-GaAs/Aspnes-9.9.yml\' database_matminer/Al99Ga901As1000/")

os.mkdir("database_matminer/Al2329O2612N588")
os.mkdir("database_matminer/Al2351O2547N653")
os.mkdir("database_matminer/Al2356O2531N669")
os.mkdir("database_matminer/Al2372O2483N717")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlN-Al2O3/Hartnett-5.88.yml\' database_matminer/Al2329O2612N588/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlN-Al2O3/Hartnett-6.53.yml\' database_matminer/Al2351O2547N653/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlN-Al2O3/Hartnett-6.69.yml\' database_matminer/Al2356O2531N669/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlN-Al2O3/Hartnett-7.17.yml\' database_matminer/Al2372O2483N717/")

os.mkdir("database_matminer/Al10Ga90Sb100")
os.mkdir("database_matminer/Al30Ga70Sb100")
os.mkdir("database_matminer/Al50Ga50Sb100")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlSb-GaSb/Ferrini-10.yml\' database_matminer/Al10Ga90Sb100/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlSb-GaSb/Ferrini-30.yml\' database_matminer/Al30Ga70Sb100/")
os.system("cp \'database_copy/data/other/semiconductor alloys/AlSb-GaSb/Ferrini-50.yml\' database_matminer/Al50Ga50Sb100/")

os.mkdir("database_matminer/In52Ga48As100")
os.system("cp \'database_copy/data/other/semiconductor alloys/GaAs-InAs/Adachi.yml\' database_matminer/In52Ga48As100/")

os.mkdir("database_matminer/In52Ga48As24P76")
os.system("cp \'database_copy/data/other/semiconductor alloys/GaAs-InAs-GaP-InP/Adachi.yml\' database_matminer/In52Ga48As24P76/")

os.mkdir("database_matminer/Ga51In49P100")
os.system("cp \'database_copy/data/other/semiconductor alloys/GaP-InP/Schubert.yml\' database_matminer/Ga51In49P100/")

os.mkdir("database_matminer/Si11Ge89")
os.mkdir("database_matminer/Si20Ge80")
os.mkdir("database_matminer/Si28Ge72")
os.mkdir("database_matminer/Si47Ge53")
os.mkdir("database_matminer/Si48Ge52")
os.mkdir("database_matminer/Si65Ge35")
os.mkdir("database_matminer/Si85Ge15")
os.mkdir("database_matminer/Si98Ge2")
os.system("cp \'database_copy/data/other/semiconductor alloys/Si-Ge/Jellison-11.yml\' database_matminer/Si11Ge89/")
os.system("cp \'database_copy/data/other/semiconductor alloys/Si-Ge/Jellison-20.yml\' database_matminer/Si20Ge80/")
os.system("cp \'database_copy/data/other/semiconductor alloys/Si-Ge/Jellison-28.yml\' database_matminer/Si28Ge72/")
os.system("cp \'database_copy/data/other/semiconductor alloys/Si-Ge/Jellison-47.yml\' database_matminer/Si47Ge53/")
os.system("cp \'database_copy/data/other/semiconductor alloys/Si-Ge/Jellison-48.yml\' database_matminer/Si48Ge52/")
os.system("cp \'database_copy/data/other/semiconductor alloys/Si-Ge/Jellison-65.yml\' database_matminer/Si65Ge35/")
os.system("cp \'database_copy/data/other/semiconductor alloys/Si-Ge/Jellison-85.yml\' database_matminer/Si85Ge15/")
os.system("cp \'database_copy/data/other/semiconductor alloys/Si-Ge/Jellison-98.yml\' database_matminer/Si98Ge2/")

# Remove unncessary folders and files
os.system("rm -r database_copy")
