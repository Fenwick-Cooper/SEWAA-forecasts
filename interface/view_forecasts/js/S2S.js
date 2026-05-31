// Code for the S2S forecast plots

let regionName = "Nairobi"
let maxRain = 30;				// Rainfall threshold in mm/week
let probability = 0.95;			// Between 0 and 1

// Hack to force reloading of dates files
let dateLoadNumber = Math.floor(Math.random() * 10000);

let availableDates;				// An object containing the initialisation dates we can use
let S2SForecast = [];			// An array of countsData objects

// Called by the regionSelect menu
async function regionSelect() {
	regionName = document.getElementById("regionSelect").value;
	// XXX Not needed? await loadForecast();
	drawPlots();
}

// Called by the initYearSelect, initMonthSelect,initDaySelect and initTimeSelect menus
async function initTimeSelect() {
	updateDateMenus();
	await loadForecast();
	drawPlots();
}

// Called by the validTimeSelect menu
async function validTimeSelect() {
	await loadForecast();
	drawPlots();
}

// Called when the focus is lost in the value threshold input
function pValueThresholdInput() {
	drawPlots();
}

// Called when the focus is lost in the probability threshold input
function pProbabilityThresholdInput() {
	drawPlots();
}

// Loads and plots the currently selected forecast
async function loadForecast() {
	let year = document.getElementById("initYearSelect").value;
	let month = document.getElementById("initMonthSelect").value;
	let day = document.getElementById("initDaySelect").value;
	let validTimeMenu = document.getElementById("validTimeSelect").value;
	
	// The directory name depends upon which model we are looking at
	let countsDir = "../data/counts_s2s";	
	
	let fileName = countsDir+"/"+year+"/"+year+"-"
								 +month.padStart(2,'0')+"-"
								 +day.padStart(2,'0')
								 +"_histogram_admin1_merged_KeEtRwUg_"
								 +validTimeMenu+"wklead.nc";
									 
	// Load data into the forecastDataObject
	let accumulationHours = 24*7;
	let modelName = "7 day accumulation";
	await S2SForecast[0].loadS2SForecast(fileName, modelName, accumulationHours);
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
	
	// Add the "Plot all valid times" menu item.
	if (datesText.length > dateObject.length) {
		if ((datesText[datesText.length-1] == "Plot all valid times") ||
			(datesText[datesText.length-1] == "Plot all initialisation times")) {
			let option = document.createElement("option");
			option.value = "All";
			option.innerHTML = datesText[datesText.length-1];
			dateSelect.appendChild(option);
		}
	}
	
	if (date != "All") {
		// If the specified year/month/day/time/valid time does not exist.
		if (!(dates.includes(date))) {
			date = dates[dates.length-1];	// Pick the final one
		}
	}
	
	// Set the menu to the value it should be
	dateSelect.value = date;
	
	// Return the value set
	return date;
}

function updateDateMenus() {
	// Are we using initialisation or valid dates for the dates menus?
//	let initDateUsed = (document.getElementById("initialisationSelect").value == "initialisationDate");
	let initDateUsed = true;
	let menuDates;
	let offsetSign;
	if (initDateUsed) {
		menuDates = availableDates;
		offsetSign = "+";
	} else {
		menuDates = validDates;
		offsetSign = "-";
	}
	
	// The available months are listed in menuDates
	year = updateMenu(menuDates,[],"initYearSelect");
	
	// The available months depend upon the year
	let yearObject = menuDates[String(year)];
	month = updateMenu(yearObject,[],"initMonthSelect");
	
	// The available days depend upon the year and month
	let monthObject = yearObject[String(month)];
	day = updateMenu(monthObject,[],"initDaySelect");
	
	// The available times depend upon the year, month and day
// 	let daysObject = monthObject[String(day)];
// 	// We use a custom string for the time menu elements
// 	let times = Object.keys(daysObject);
// 	let timeStrings = new Array(times.length);
// 	for (let i=0;i<times.length;i++) {
// 		timeStrings[i] = times[i].padStart(2,'0')+":00 UTC";
// 	}
// 	time = updateMenu(daysObject,timeStrings,"initTimeSelect");
	
	// The available valid times depend upon the year, month, day and time.
	validTimes = monthObject[String(day)];	// validTimes is an Array
	// We use a custom string for the valid time menu elements
	let validTimeStrings = new Array(validTimes.length+1);
	for (let i=0;i<validTimes.length;i++) {
		if (initDateUsed) {
			// What's the valid date? (YYYY-MM-DD)
			validDate = timeOffsetToDate((validTimes[i]-1)*24*7,
										 year+"-"+String(month).padStart(2,'0')
											 +"-"+String(day).padStart(2,'0'));
		} else {
			// What's the initialisation date? (YYYY-MM-DD)
			validDate = timeOffsetToDate(-validTimes[i],
										 year+"-"+String(month).padStart(2,'0')
											 +"-"+String(day).padStart(2,'0'));
		}
				
		validTimeStrings[i] = validDate.getUTCFullYear()
							+"-"+String(validDate.getUTCMonth()+1).padStart(2,'0')
							+"-"+String(validDate.getUTCDate()).padStart(2,'0')
							+" "+String(validDate.getUTCHours()).padStart(2,'0')
							+":00 UTC ("+offsetSign+(validTimes[i]-1)+" weeks)";
	}
	// Add an "Plot all valid times" option
// 	if (initDateUsed) {
// 		validTimeStrings[validTimes.length] = "Plot all valid times";
// 	} else {
// 		validTimeStrings[validTimes.length] = "Plot all initialisation times";
// 	}
 	updateMenu(validTimes,validTimeStrings,"validTimeSelect");
}

async function loadDates() {
	// Fetch a remote file
	let fileName = "../data/counts_s2s/available_dates.json?"+dateLoadNumber;

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

function initControls() {
	// XXX Actually should keep the settings on reload and reload the correct plots
	
	document.getElementById("regionSelect").value = regionName;
	
	// Need to get the units correct
	document.getElementById("thresholdValueSelect").value = roundSF(maxRain, 3);
	document.getElementById("thresholdProbabilitySelect").value = roundSF((probability*100), 4);
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
	
	// Set the default values of the plot controls
	initControls();
	
	// Specify the function to call to inform the user what is going on
	setStatusUpdateFunction(showLoadingStatus);
	
	// GANForecast is a global array of countsData objects
	S2SForecast[0] = new s2sCountsData();		// Create a countsData object
	
	// Load a list of the available forecasts
	await loadDates();
	
	// Load the currently selected forecast
	await loadForecast();
	
	// Draw everything
	await drawPlots();
	
	// Detect if the enter or return key is pressed in the document
	document.addEventListener("keydown", function(event) {
		if (event.key === "Enter") {
			drawPlots();
		}
	});
}

async function drawPlots() {
	// XXX Do I need all of these globals?
	
	// It's easier to update the plot explanation every time the plots are drawn
	// showExplanation(); XXX
	
	let units = "mm/week";
	//let region = "Nairobi"  // XXX Get from menu
	let x = 2, y=2;			// Location of plot from top left
	let width = 500;		// Width of plot in pixels
	let height = 500;		// Height of plot in pixels
	
	// See what the input boxes say
	let norm = 1;
	maxRain = document.getElementById("thresholdValueSelect").value / norm;
	probability = document.getElementById("thresholdProbabilitySelect").value / 100.0;
	
	// Find out how many plots to make
	let plotAllValidTimes = document.getElementById("validTimeSelect").value;
	if (plotAllValidTimes == "All") {
		numCanvases = validTimes.length;
	} else {
		numCanvases = 1;
	}
	
	// Ensure the correct number of histogram canvases exist
	let canvasNum=0;
	while (document.getElementById("histogramCanvas"+canvasNum) != null) {
		// If this canvas is not needed
		if (canvasNum+1 > numCanvases) {
			canvasElement = document.getElementById("histogramCanvas"+canvasNum);
			canvasElement.remove();
		}
		canvasNum += 1;
	}
	// If there are insufficient canvases
	brIdx=0;	// Keep track of the number of line breaks
	while (canvasNum < numCanvases) {
		const canvasElement = document.createElement("canvas");
		canvasElement.id = "histogramCanvas"+canvasNum;
		canvasElement.width = 511;
		canvasElement.height = 504;
		canvasElement.innerHTML = "Your browser does not support the HTML canvas tag.";
		// canvasElement.style="border:1px solid grey";
		
		// Place the histogram canvas just after the statusText
		const statusElement = document.getElementById("statusText");
		statusElement.insertAdjacentElement("afterend", canvasElement);
					
		canvasNum += 1;
	}

	// Draw plots in each canvas
	for (let canvasNum=0;canvasNum<numCanvases;canvasNum++) {
			
		// Create a new histogram specification
		let barChartSpec = new barChartSpecification();
		
		// Get the context for plotting
		const histogramCanvas = document.getElementById("histogramCanvas"+canvasNum);
		const histogramCtx = histogramCanvas.getContext("2d");
		
		// Erase the canvas
		histogramCtx.clearRect(0,0,histogramCanvas.width,histogramCanvas.height);
		
		// Plot the histogram and wait for it to finish
		await S2SForecast[canvasNum].plotHistogram(histogramCtx, x, y, width, height,
					maxRain, probability, regionName, units, barChartSpec);
	}
}

init();