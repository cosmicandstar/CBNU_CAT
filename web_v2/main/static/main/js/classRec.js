

window.onload = function () {
    changeClassType();
}

function moreshow(id) {
    var moreInfoPage = document.getElementById("moreinfo_" + id);
    if (moreInfoPage.style.display == 'none') {
        moreInfoPage.style.display = 'block';
    } else {
        moreInfoPage.style.display = 'none'
    }
}

function changeClassType() {

    var classSelector = document.querySelector(".classTypeSelect1");
    var selectedClass = classSelector.options[classSelector.selectedIndex].value;
    var selectedName = classSelector.options[classSelector.selectedIndex].text;
    var selectors = document.querySelector(".classTypeFrame");
    document.getElementById("menu2").style.display = 'none';
    document.getElementById("menu3").style.display = 'none';
    document.getElementById("menu4").style.display = 'none';
    document.getElementById("menu5").style.display = 'none';
    if ( selectedClass == 0 ) {
        document.getElementById("menu2").style.display = 'inline-block';
    }
    if ( selectedClass == 1 ) {
        document.getElementById("menu3").style.display = 'inline-block';
    }
    if ( selectedClass == 2 ){
        document.getElementById("menu4").style.display = 'inline-block';
    }
    if ( selectedClass == 3 ){
        document.getElementById("menu5").style.display = 'inline-block';
    }
}

function classAdd(id) {
    var classAddSetting = document.getElementById("classAddSetting_" + id);
    if (classAddSetting.style.display == 'none') {
        classAddSetting.style.display = 'block';
    } else {
        classAddSetting.style.display = 'none'
    }
}