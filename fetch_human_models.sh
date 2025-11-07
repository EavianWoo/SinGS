# this is a script to download SMPL-X and FLAME assets
# Make sure to register and login https://smpl-x.is.tue.mpg.de/ & https://flame.is.tue.mpg.de/

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }


PROJ_DIR=$(pwd)
DOWNLOAD_PATH="./data/human_models"
mkdir -p ${DOWNLOAD_PATH}
echo -e "\nTo download files, you need to first register https://smplify.is.tue.mpg.de and https://mano.is.tue.mpg.de, then you can log in with the script."
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password

username=$(urle $username)
password=$(urle $password)

# [SMPL]
# file to dowload:
# - basicModel_neutral_lbs_10_207_0_v1.0.0.pkl (or basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl)
#   https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip
#   or 
#   https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip
# download manually if you need
# then rename to SMPL_NEUTRAL.pkl
mkdir ${DOWNLOAD_PATH}/smpl
{
    wget --post-data "username=$username&password=$password" \
        'https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip' \
        -O ${DOWNLOAD_PATH}/smplify.zip --no-check-certificate --continue
} &&  unzip ${DOWNLOAD_PATH}/smplify.zip -d ${DOWNLOAD_PATH}/

mv ${DOWNLOAD_PATH}/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl ${DOWNLOAD_PATH}/smpl/SMPL_NEUTRAL.pkl
rm -rf ${DOWNLOAD_PATH}/smplify.zip
rm -rf ${DOWNLOAD_PATH}/smplify_public


# [SMPL-H]
# files to dowload:
# - SMPLH_NEUTRAL.pkl
#   https://huggingface.co/lithiumice/models_hub/resolve/a1004b6b826279d04634cbbf8fdd1879e7503fc9/smpl_smplh_smplx_mano/SMPLH_NEUTRAL.pkl \
#
# - SMPLH_FEMALE.pkl, SMPLH_MALE.pkl
#   https://download.is.tue.mpg.de/download.php?domain=mano&sfile
# download manually if you need
mkdir ${DOWNLOAD_PATH}/smplh
{
    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=mano&sfile=smplx.zip' -O ${DOWNLOAD_PATH}/smplx.zip --no-check-certificate --continue
} && unzip ${DOWNLOAD_PATH}/smplx.zip -d ${DOWNLOAD_PATH}/

mv ${DOWNLOAD_PATH}/smplx/smplh/* ${DOWNLOAD_PATH}/smplh/
rm -rf ${DOWNLOAD_PATH}/smplx.zip
rm -rf ${DOWNLOAD_PATH}/smplx

wget -c https://huggingface.co/lithiumice/models_hub/resolve/a1004b6b826279d04634cbbf8fdd1879e7503fc9/smpl_smplh_smplx_mano/SMPLH_NEUTRAL.pkl \
    -O ${DOWNLOAD_PATH}/smplh/SMPLH_NEUTRAL.pkl

# # SMPLX Models
# {
#     wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O ${DOWNLOAD_PATH}/models_smplx_v1_1.zip --no-check-certificate --continue
# } && unzip ${DOWNLOAD_PATH}/models_smplx_v1_1.zip -d ${DOWNLOAD_PATH}/

