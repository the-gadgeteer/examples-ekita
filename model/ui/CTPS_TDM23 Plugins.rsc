
Macro "Model.Attributes" (Args,Result)
    Attributes = {
        {"BackgroundColor",{64,64,64}},
        {"BannerHeight", 90},
        {"BannerWidth", 960},
        {"Base Scenario Name", "Base"},
        {"ClearLogFiles", 1},
        {"CloseOpenFiles", 1},
        {"ResizeImage", 1},
        {"CodeUI", "ui\\tdm23_ui.dbd"},
        {"DebugMode", 1},
        {"DrawParameterTypes", "All Parameter Types"},
        {"ExpandStages", "Side by Side"},
        {"HideBanner", null},
        {"Layout", null},
        {"MaxEngines", 2},
        {"MaxProgressBars", 2},
        {"MinItemSpacing", 5},
        {"Output Folder Format", "OutputFolder\\Scenario Name"},
        {"Output Folder Parameter", "OutputFolder"},
        {"Output Folder Per Run", "by Scenario Name"},
        {"Picture", "ui\\ctps.bmp"},
        {"ReportAfterStep", 0},
        {"ResizeImage", 1},
        {"RunParallel", 0},
        {"Requires",
            {{"Program", "TransCAD"},
            {"Version", 9},
            {"Build", 32775}}},
        {"Shape", "Rectangle"},
        {"SourceMacro", "Model.Attributes"},
        {"Time Stamp Format", "yyyyMMdd_HHmm"},
        {"ShowTaskMonitor", 1}
    }
EndMacro


Macro "Model.Step" (Args,Result)
    Attributes = {
        {"FillColor",{255,255,255}},
        {"FillColor2",{192,192,125}},
        {"FrameColor",{255,255,255}},
        {"Height", 35},
        {"PicturePosition", "CenterRight"},
        {"TextColor",{0,0,0}},
        {"TextFont", "Calibri|11|400|000000|0"},
        {"Width", 225}
    }
EndMacro


Macro "Model.Arrow" (Args,Result)
    Attributes = {
        {"ArrowHead", "Triangle"},
        {"ArrowHeadSize", 6},
        {"Color", "#ff8000"},
        {"FeedbackColor", "#ff4000"},
        {"FillColor", "#ff8000"},
        {"ForwardColor", "#ffc000"},
        {"PenWidth", 1},
        {"TextColor", "#202020"},
        {"TextStyle", "Center"},
        {"ArrowBase", "No Arrow Head"}
    }
EndMacro


Macro "Model.OnModelReady" (Args,Result)
Body:
    Return({"Base Folder": "%Model Folder%"})
EndMacro


Macro "Model.OnModelLoad" (Args, Results)
Body:
    flowchart = RunMacro("GetFlowChart")
    { drive , path , name , ext } = SplitPath(flowchart.UI)
    rootFolder = drive + path
 
    ui_DB = rootFolder + "UI\\tdm23_ui.dbd"
    srcFile = rootFolder + "Code\\model\\gisdk\\compile_ui.lst"
    RunMacro("CompileGISDKCode", {Source: srcFile, UIDB: ui_DB, Silent: 0, ErrorMessage: "Error compiling TDM23 code"})

    if lower(GetMapUnits()) <> "miles" then
        MessageBox("Set the system units to miles before running the model", {Caption: "Warning", Icon: "Warning", Buttons: "yes"})
    return(true)
EndMacro


// do not run the model if the map units are incorrect
Macro "Model.CanRun" (args)
Body:
    retStatus = true
    currMapUnits = GetMapUnits()
    if lower(currMapUnits) <> "miles" then do
        retStatus = false
        msgText = Printf("Current map units are '%s'. Please change to 'Miles' before running the model.", {currMapUnits})
        MessageBox( msgText, { Caption: "Error", Buttons: "OK", Icon: "Error" })
    end
    return(retStatus)
EndMacro


Macro "Model.OnModelStart" (Args,Result,StepName)
Body:

    out_folder = Args.[OutputFolder]
    out_folder = out_folder + "\\"

    // check for existence of required subfolder names. If they don't exist, create them
    dirs = {"_demand","_assignment","_assignment\\pnr","_skim","_networks","_summary","_logs","_feedback",
            "_demand\\mc","_demand\\td","_demand\\tod","_demand\\td\\modelfiles", "_demand\\mc\\modelfiles", 
             "_summary\\tld", "_summary\\geo", "_summary\\trips", "_summary\\zonal", "_summary\\hwy", "_summary\\trn",
             "_summary\\postproc", "_summary\\emat",
             "_postproc","_postproc\\airquality","_postproc\\equity"}
    
    folder = {out_folder} + dirs.map(do (f) return(out_folder + f) end)
    for f in folder do
        o = CreateObject("CC.Directory", RunMacro("FlowChart.ResolveValue", f, Args))
        o.Create()
    end

    // initialize the model python class and assign it to the list of args 
    mr = CreateObject("Model.Runtime")
    pyObj = mr.RunCode("InitializePYClass", mr.GetValues())

    // set scenario name
    {scenariofile, scen_path} = mr.GetScenario() // for current scenario
    scen_arr = ParseString(scen_path,"\\")
    scen_name = scen_arr[ArrayLength(scen_arr)] // last element is scenario name
    Args.Scenario = scen_name

    return({PY: pyObj})
EndMacro


Macro "Model.OnModelDone" (Args,Result,StepName)
Body:
    out_dir              = Args.[Output Folder]
    rep_file = out_dir + "\\_summary\\" + Args.Scenario + "_report.html"

    html = RunMacro("ConvertXML_to_HTML", {XML: out_dir + "\\CTPS_TDM23-Run-Report.xml", XmlType: "Report", OutFileName: rep_file}) // Report, Summary - Log, ErrorLog
    LaunchDocument(html,)
    return(true)
EndMacro

