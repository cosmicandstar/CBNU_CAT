

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


function moreshow() {
    var moreInfoPage = document.getElementById("more_info_");
    if (moreInfoPage.style.display == 'none') {
        moreInfoPage.style.display = 'block';
    } else {
        moreInfoPage.style.display = 'none'
    }
}
