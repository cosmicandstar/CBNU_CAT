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

    var mymbti = document.querySelector(".mymbti");
    mymbti.innerHTML = selectedMbti;

    var hidden_mymbti = document.querySelector(".hidden_mymbti");
    hidden_mymbti.value = selectedMbti;

    var mbtiframe = document.querySelector(".mbtiframe");
    if(selectedMbti === "INFJ" || selectedMbti === "INFP" ||
        selectedMbti === "ENFJ" || selectedMbti === "ENFP" ){
        mbtiframe.style.background = "linear-gradient(-45deg, #a8e063, #56ab2f)";
        //green
    }
    else if(selectedMbti === "INTJ" || selectedMbti === "INTP" ||
        selectedMbti === "ENTJ" || selectedMbti === "ENTP" ){
        mbtiframe.style.background = "linear-gradient(-45deg,rgb(247, 162, 187), rgb(221, 33, 74))";
        //red
    }
    else if(selectedMbti === "ISFJ" || selectedMbti === "ISTJ" ||
        selectedMbti === "ESFJ" || selectedMbti === "ESTJ" ){
        mbtiframe.style.background = "linear-gradient(-45deg, #56CCF2, #2F80ED)";
        //blue
    }
    else if(selectedMbti === "ISFP" || selectedMbti === "ISTP" ||
        selectedMbti === "ESTP" || selectedMbti === "ESFP" ){
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
