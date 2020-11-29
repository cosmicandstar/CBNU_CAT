//mbti change
var mbtiframe = document.querySelector(".mbtiframe");

window.onload = function () {
    var mbtiselector = document.querySelector(".mbtiSelect").options;
    var hidden_mymbti = document.querySelector(".hidden_mymbti");
    var cnt = mbtiselector.length;
    for (var i = 0; i < cnt; i++) {
        console.log(mbtiselector[i].value, hidden_mymbti.value);
        if (mbtiselector[i].value == hidden_mymbti.value) {
            mbtiselector[i].selected = true;
            break;
        }
    }
    changeMbti(true)
}

function changeMbti(first) {
    var mbtiselector = document.querySelector(".mbtiSelect");
    var selectedMbti = mbtiselector.options[mbtiselector.selectedIndex].value;
    var mbti = {
        "0": "INTJ",
        "1": "INTP",
        "2": "INFP",
        "3": "INFP",
        "4": "ISTJ",
        "5": "ISTP",
        "6": "ISFJ",
        "7": "ISFP",
        "8": "ENTJ",
        "9": "ENTP",
        "10": "ENFJ",
        "11": "ENFP",
        "12": "ESTJ",
        "13": "ESTP",
        "14": "ESFJ",
        "15": "ESFP",
    };
    var mymbti = document.querySelector(".mymbti");


    mymbti.innerHTML = mbti[selectedMbti];

    var hidden_mymbti = document.querySelector(".hidden_mymbti");
    hidden_mymbti.value = selectedMbti;

    var mbtiframe = document.querySelector(".mbtiframe");
    if(selectedMbti === "2" || selectedMbti === "3" ||
        selectedMbti === "10" || selectedMbti === "11" ){
        mbtiframe.style.background = "linear-gradient(-45deg, #a8e063, #56ab2f)";
        //green
    }
    else if(selectedMbti === "0" || selectedMbti === "1" ||
        selectedMbti === "8" || selectedMbti === "9" ){
        mbtiframe.style.background = "linear-gradient(-45deg,rgb(247, 162, 187), rgb(221, 33, 74))";
        //red
    }
    else if(selectedMbti === "6" || selectedMbti === "4" ||
        selectedMbti === "14" || selectedMbti === "12" ){
        mbtiframe.style.background = "linear-gradient(-45deg, #56CCF2, #2F80ED)";
        //blue
    }
    else if(selectedMbti === "7" || selectedMbti === "5" ||
        selectedMbti === "13" || selectedMbti === "15" ) {
        mbtiframe.style.background = "linear-gradient(-45deg, #8E2DE2, #4A00E0)";
        //purple
    }
    if (!first)
        document.getElementById("mbti_form").submit();
}

/////////////////////
//change class type
function changeClassType(){
    var classTypeSelector = document.querySelector(".classTypeSelect");
    var selectedclassType = classTypeSelector.options[classTypeSelector.selectedIndex].value;
    console.log(selectedclassType);

    var myClassType = document.querySelector(".myClassType");
    myClassType.innerHTML = selectedclassType;
}



/////////////////////
//keyword add, delete
//var keywordframe = document.querySelector(".keywordframe");
var keywordlist = ["조별과제" , "필기중요" , "이론 중심 수업" , "수업시간 엄수" , "매주 리포트" , "발표수업"]
function addingkword(word){
    var worddiv = document.createElement('div');
    var delbt = document.createElement('button');
    worddiv.innerText = word;
    worddiv.class = "mykeyword";
    delbt.innerText = "X";
    worddiv.appendChild(delbt);
    console.log('div added');
    console.log('word added: ' , word);
    console.log(word);

    var add = document.querySelector(".keyword");
    add.appendChild( worddiv );
}

function keywordAddClick(word){
    addingkword(word);
    console.log("add");
}


function moreshow(id) {
    var moreInfoPage = document.getElementById("more_info_" + id);
    if (moreInfoPage.style.display == 'none') {
        moreInfoPage.style.display = 'block';
    } else {
        moreInfoPage.style.display = 'none'
    }
}
