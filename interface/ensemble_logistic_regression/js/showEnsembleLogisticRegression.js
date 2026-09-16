// Javascript for plotting the results of the ensemble logistic regression.

// Hack to force reloading of dates files
let dateLoadNumber = Math.floor(Math.random() * 10000);
let availableDates;				// An object containing the dates we can use

let ELRForecast = new ELRData();		// Create a global ELRData object

// Select a menu item requiring data to be loaded
async function ELRSelect() {
	let region = document.getElementById("regionSelect").value;
	
	// Each region has a different canvas size
	let width,height;
	if (region == "Kenya") {
		width = 605;
		height = 770;
	} else if (region == "Ethiopia") {
		width = 1130;
		height = 860;
	} else if (region == "Uganda") {
		width = 630;
		height = 656;
	}  else if (region == "Rwanda") {
		width = 690;
		height = 596;
	}
	
	// Set the canvas size
	document.getElementById("ClimateCanvas").width = width;
	document.getElementById("ClimateCanvas").height = height;
	document.getElementById("ELRCanvas").width = width;
	document.getElementById("ELRCanvas").height = height;
	
	// Load the dates that are available for this country
	await loadDates();
	
	// Load the ELR forecast
	await loadELRForecast();
	
	// Draw the ELR forecast
	await drawPlot();
}

// Change the lead time to plot
async function validTimeSelect() {
	// Draw the ELR forecast
	await drawPlot();
}

// Change the style to plot
async function styleSelect() {
	// Draw the ELR forecast
	await drawPlot();
}

// Change the threshold to plot
async function thresholdSelect() {
	
	// Load the ELR climate for the selected threshold
	await loadELRClimate();
	
	// Draw the ELR forecast
	await drawPlot();
}

// Set the threshold menu
function updateThresholdMenu(thresholds) {
	
	// Select the HTML select menu that we are updating
	let thresholdsSelect = document.getElementById("thresholdSelect");
	
	// Record the menu's value before we remove it
	let threshold = thresholdsSelect.value;
	
	// Remove all of the current menu items
	while (thresholdsSelect.hasChildNodes()) {
		thresholdsSelect.removeChild(thresholdsSelect.firstChild);
	}
	
	// Add the appropriate menu items
	let option;
	for (let i=0; i<thresholds.length;i++) {
		option = document.createElement("option");
		option.value = thresholds[i];
		option.innerHTML = thresholds[i];
		thresholdsSelect.appendChild(option);
	}
	
	// If the original value does not exist.
	if (!(thresholds.includes(threshold))) {
		threshold = thresholds[thresholds.length-1];	// Pick the final one
	}
	
	// Set the menu to the value it should be
	thresholdsSelect.value = threshold;
}

// When the user focus leaves a probability bin edge input
function pBinInput() {
	// Draw the ELR forecast
	drawPlot();
}

// Has an alert been called yet
alertCalledG = false;

// Parse and return the probability bin edges input boxes
async function getPBinEdges() {

	let inputValue;
	let pBinEdges = [];
	
	pBinEdges[0] = 0;
	inputValue = document.getElementById("probabilityBin1").value;
	pBinEdges[1] = parseInt(inputValue);
	inputValue = document.getElementById("probabilityBin2").value;
	pBinEdges[2] = parseInt(inputValue);
	inputValue = document.getElementById("probabilityBin3").value;
	pBinEdges[3] = parseInt(inputValue);
	inputValue = document.getElementById("probabilityBin4").value;
	pBinEdges[4] = parseInt(inputValue);
	pBinEdges[5] = 100;
	
	// Add alert for parse errors
	let err = false;
	for (let i=1;i<pBinEdges.length;i++) {
		
		if (!(pBinEdges[i] >= pBinEdges[i-1])) {
			pBinEdges[i] = pBinEdges[i-1];
			err = true;
		} else if (pBinEdges[i] < 0) {
			pBinEdges[i] = 0;
			err = true;
		} else if (pBinEdges[i] > 100) {
			pBinEdges[i] = 100;
			err = true;
		}
		
		if (err) {
			document.activeElement.blur();
			if (!alertCalledG) {
				alertCalledG = true;
				await alert("Probability bins must be increasing (left to right) and be between 0 and 100.");
			}
			break;
		}
	}
	
	// Now set the boxes to the integer value (removes surplus rubbish)
	document.getElementById("probabilityBin1").value = pBinEdges[1];
	document.getElementById("probabilityBin2").value = pBinEdges[2];
	document.getElementById("probabilityBin3").value = pBinEdges[3];
	document.getElementById("probabilityBin4").value = pBinEdges[4];
	
	alertCalledG = false;
	return pBinEdges;
}

// XXX Needs to be called by ELRSelect
async function loadDates() {

	// Which country are we looking at
	regionName = document.getElementById("regionSelect").value;

	// Fetch a remote file
	let fileName;
	// XXX Desparate hack
	if (regionName == "Rwanda") {
		fileName = "../ELR_predictions/24h_accumulations/"+regionName+"/county/available_dates.json?"+dateLoadNumber;
	} else {
		fileName = "../ELR_predictions/24h_accumulations/"+regionName+"/subcounty/available_dates.json?"+dateLoadNumber;
	}
	
	// dateLoadNumber ensures that the available_dates.json file is not cached
	dateLoadNumber += 1;
	if (dateLoadNumber > 10000) {
		dateLoadNumber = 0;
	}
	const response = await fetch(fileName);
	
	// Parse the JSON arrayBuffer of the file and return the resulting object
	availableDates = await response.json();

	updateDateMenus();
}

// Update an HTML select menu with dates that are available.
// dateObject - Can be a years, months, day or time object (from the global
//              availableDates).
// dateText   - An array containing the menu item strings to use. If empty the dateObject
//              keys are used as menu items.
// id         - The id of the select menu element in the HTML.
function updateMenu(dateObject,datesText,id) {
	
	let dates;
	if (dateObject instanceof Array) {
		// dateObject is an array of numbers. Convert it to an array of strings
		dates = new Array(dateObject.length);
		for (let i=0;i<dates.length;i++) {
			dates[i] = String(dateObject[i]);
		}
	} else {
		// dateObject's keys are a list of strings
		dates = Object.keys(dateObject);
	}
	
	// Select the HTML select menu that we are updating
	let dateSelect = document.getElementById(id);
	
	// Record the menu's value before we remove it
	let date = dateSelect.value;
	
	// Remove all of the current menu items
	while (dateSelect.hasChildNodes()) {
		dateSelect.removeChild(dateSelect.firstChild);
	}
	
	// Add the menu items specified in dates
	for (let i=0;i<dates.length;i++) {
		let option = document.createElement("option");
		option.value = dates[i];
		if (datesText.length > 0) {
			option.innerHTML = datesText[i];
		} else {
			option.innerHTML = dates[i];
		}
		dateSelect.appendChild(option);
	}

	// If the specified year/month/day/time/valid time does not exist.
	if (!(dates.includes(date))) {
		date = dates[dates.length-1];	// Pick the final one
	}
	
	// Set the menu to the value it should be
	dateSelect.value = date;
	
	// Return the value set
	return date;
}

function updateDateMenus() {
	
	// The available months are listed in availableDates
	year = updateMenu(availableDates,[],"initYearSelect");
	
	// The available months depend upon the year
	let yearObject = availableDates[String(year)];
	month = updateMenu(yearObject,[],"initMonthSelect");
	
	// The available days depend upon the year and month
	let monthObject = yearObject[String(month)];
	day = updateMenu(monthObject,[],"initDaySelect");
	
	// The available valid times depend upon the year, month, day and time.
	validTimes = monthObject[String(day)];	// validTimes is an Array
	// We use a custom string for the valid time menu elements
	let validTimeStrings = new Array(validTimes.length+1);
	for (let i=0;i<validTimes.length;i++) {
		// What's the valid date? (YYYY-MM-DD)
		// validDate = timeOffsetToDate(validTimes[i],
		// XXX Hack for now (always a lead time of 30 hours)
		validDate = timeOffsetToDate(30,
									 year+"-"+String(month).padStart(2,'0')
										 +"-"+String(day).padStart(2,'0'));
				
		validTimeStrings[i] = validDate.getUTCFullYear()
							+"-"+String(validDate.getUTCMonth()+1).padStart(2,'0')
							+"-"+String(validDate.getUTCDate()).padStart(2,'0')
							+" "+String(validDate.getUTCHours()).padStart(2,'0')
							+":00 UTC (+"+30+"h)";
		//					+":00 UTC (+"+validTimes[i]+"h)";
	}
	updateMenu(validTimes,validTimeStrings,"validTimeSelect");
}

// Loads the currently selected ELR forecast (all lead times and thresholds)
async function loadELRForecast() {
	let region = document.getElementById("regionSelect").value;
	let year = document.getElementById("initYearSelect").value;
	let month = document.getElementById("initMonthSelect").value;
	let day = document.getElementById("initDaySelect").value;
					
	let fileName;
	// XXX Desparate hack
	if (region == "Rwanda") {
		fileName = "../ELR_predictions/24h_accumulations/"+region+"/county/GAN_"
					+year+month.padStart(2,'0')+day.padStart(2,'0')+"_ELR_v1.nc";
	} else {
		fileName = "../ELR_predictions/24h_accumulations/"+region+"/subcounty/GAN_"
					+year+month.padStart(2,'0')+day.padStart(2,'0')+"_ELR_v1.nc";
	}
	
	// Load data into the ELRDataObject
	await ELRForecast.loadELRForecast(fileName,year,month,day);
	
	// Update the threshold menu depending upon the available thresholds
	const thresholds = [];
	for (let i=0;i<ELRForecast.thresholds.length;i++) {
		thresholds[i] = ELRForecast.thresholds[i].toString(10);
	}
	updateThresholdMenu(thresholds);
	
	// Load the ELR climate for the selected threshold
	await loadELRClimate();
}

// Load the ELR climate for the selected threshold
async function loadELRClimate() {
	let threshold = document.getElementById("thresholdSelect").value;
	let month = document.getElementById("initMonthSelect").value;
	
	if ((ELRForecast.currentThreshold != threshold) || (ELRForecast.climateMonth != month)){
	
		let month = document.getElementById("initMonthSelect").value;
		fileName = "../../../ELR/climatological_exceedances/clim_exc_"+threshold+"mmday_"+month+"month.nc";
		
		// Load data into the ELRClimate ELRDataObject
		await ELRForecast.loadELRClimate(fileName, threshold, month);
	}
}

async function drawPlot(){
	
	// Plot attributes
	let regionName = document.getElementById("regionSelect").value;
	let style = document.getElementById("styleSelect").value;
	let validTime = document.getElementById("validTimeSelect").value;
	let threshold = document.getElementById("thresholdSelect").value;
	
	// Kenya
	let x = 2, y=2;			// Location of plot from top left
	
	let width,height;
	if (regionName == "Kenya") {
		width = 623; 		// Width of plot in pixels  532
		height = 760;		// Height of plot in pixels  707
	} else if (regionName == "Ethiopia") {
		width = 1125; 		// Width of plot in pixels  1029
		height = 860;		// Height of plot in pixels  791
	} else if (regionName == "Uganda") {
		width = 619;
		height = 615;		// XXX WRONG 554
	} else if (regionName == "Rwanda") {
		width = 683; 		// Width of plot in pixels  570
		height = 595;		// Height of plot in pixels  480
	}
	
	// Must use await unless all of the region shape data is loaded in advance
	// otherwise the region shape data could be loaded multiple times and corrupted.
	
	// Parse the probability bin edges input boxes
	pBinEdges = await getPBinEdges();
	
	// Get the context for plotting
	let canvas = document.getElementById("ClimateCanvas");
	let ctx = canvas.getContext("2d");
	
	// Erase the canvas
	ctx.clearRect(0,0,canvas.width,canvas.height);
	
	// Draw the ELR forecast
	let plotRect = await ELRForecast.plotExceedenceClimate(ctx, x, y, width, height,
														       style, validTime, threshold, 
														       pBinEdges, regionName);
	
	// Get the context for plotting
	canvas = document.getElementById("ELRCanvas");
	ctx = canvas.getContext("2d");
	
	// Erase the canvas
	ctx.clearRect(0,0,canvas.width,canvas.height);
	
	// Draw the ELR forecast
	plotRect = await ELRForecast.plotExceedenceProbability(ctx, x, y, width, height,
														   style, validTime, threshold, 
														   pBinEdges, regionName);
}

// Function to inform the user what is going on
//    code - 0 = Not waiting
//           1 = Waiting for data to load
//           2 = Waiting for calculations
//           3 = Waiting for plots to draw
// message - A description of what we are waiting for
function showLoadingStatus(code, message) {

	if (code == 0) {	// We are not waiting
		document.getElementById("statusText").style.color = "black";
		
	} else {			// We are waiting
		document.getElementById("statusText").style.color = "#cc0000";	// dark red
	}
	
	// Inform the user what is going on
	document.getElementById("statusText").innerHTML = message;
}

async function init() {
	
	// Specify the function to call to inform the user what is going on
	setStatusUpdateFunction(showLoadingStatus);
	
	// Set appropriate menus
	await ELRSelect();
	
	// Load a list of the available forecasts
	await loadDates();
	
	// Load the currently selected forecast
	await loadELRForecast();
	
	// Detect if the enter or return key is pressed in the document
	document.addEventListener("keydown", function(event) {
		if (event.key === "Enter") {
			drawPlot();
		}
	}); 
	
	// Draw the ELR forecast
	await drawPlot();
}

init();
