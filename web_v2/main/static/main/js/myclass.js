function moreshow(id) {
    var moreInfoPage = document.getElementById("moreinfo_" + id);
    if (moreInfoPage.style.display == 'none') {
        moreInfoPage.style.display = 'block';
    } else {
        moreInfoPage.style.display = 'none'
    }
}


function classReset(id) {
    var classAddSetting = document.getElementById("classReSetting_" + id);
    if (classAddSetting.style.display == 'none') {
        classAddSetting.style.display = 'block';
    } else {
        classAddSetting.style.display = 'none'
    }
}