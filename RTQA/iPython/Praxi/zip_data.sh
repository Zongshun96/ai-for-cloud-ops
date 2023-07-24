set -e
# backup data
for d in ./data*/ ;
do
    echo $d "${d%/}.zip"
    zip -r "${d%/}.zip" "$d" >/dev/null
    # swift --os-auth-type v3applicationcredential \
    # --os-application-credential-id 2e4e810fa6ac4afe8179dc4d4b706a45 \
    # --os-application-credential-secret dVtjOXp4Hr2mC2oKRDcvCnHsS-EBy_TyJm907VFz1APHAvlv0suBYE7d6_UURZe_QXgEFj6ZYdwPHKLblG2Uew \
    # upload --changed --segment-size 4831838208 \
    # FSL_Journal-05032023 "${d%/}.zip"
    # rm "${d%/}.zip"
    # rm -fr "$d"
    # # break
done