
Class "Visualize.Menu.Items"
EndClass


Macro "OpenParamFile" (Args,Result)
Body:
	mr = CreateObject("Model.Runtime")
	curr_param = mr.GetSelectedParamInfo()
	result = mr.OpenFile(curr_param.Name)
EndMacro


MenuItem "CTPS_TDM23 Menu Item" text: "CTPS_TDM23"
    menu "CTPS_TDM23 Menu"

menu "CTPS_TDM23 Menu"
    init do
	runtimeObj = CreateObject("Model.Runtime")
	curr_param = runtimeObj.GetSelectedParamInfo() 
	menu_items = {"Show Map", "Show Matrix", "Show Table"}
	if curr_param = null then
		DisableItems(menu_items)
	status = curr_param.Status
	if status = "Missing" then DisableItems(menu_items)
	else if status = "Exists" then do
		type = curr_param.Type
		if type = "NETWORK" then type = "MAP"
		menu_item = "Show " + Proper(type)
		DisableItems(menu_items)
		EnableItem(menu_item)
		end

    {, scen} = runtimeObj.GetScenario()
    if scen = null then 
        DisableItem("Select Query Analysis Menu Item")
    else 
        EnableItem("Select Query Analysis Menu Item")
    EndItem // end of init

    MenuItem "Show Map" text: "Show Map"
        do 
        RunMacro("OpenParamFile")
        enditem 

    MenuItem "Show Matrix" text: "Show Matrix"
        do 
        RunMacro("OpenParamFile")
        enditem 

    MenuItem "Show Table" text: "Show Table"
        do 
        RunMacro("OpenParamFile")
        enditem 

    MenuItem "Select Query Analysis Menu Item" text: "Select Query Analysis"
        do
        mr = CreateObject("Model.Runtime") // runtimeObj cannot be accessed here 
        dbox_res = RunDbox("config query", mr) 
        ready = dbox_res[1]

        if ready = 1 then do
            Args = mr.GetValues()
            Args.[Run Select Query from Menu] = 1
            mode = dbox_res[2] // 1 for highway, 2 for transit
            qry_file = dbox_res[3]
            res_folder = dbox_res[4] 

            if mode = 1 then do
                Args.[Highway Select Query File] = qry_file 
                Args.[Highway Select Query Output Folder] = res_folder
                for tod in {"am", "md", "pm", "nt"} do
                    ok = mr.RunCode("highway_assignment", Args, tod)
                end
                if ok =1 then
                    ShowMessage("Highway Select Query Analysis finished.")
                end
            else if mode = 2 then do
                Args.[Transit Select Query File] = qry_file 
                Args.[Transit Select Query Output Folder] = res_folder

                for tod in {"am", "md", "pm", "nt"} do
                    trip_tab = Args.("Per Trips - " + tod)
                    for mode in {"ta_acc", "ta_egr", "tw"} do
                        ok = mr.RunCode("transit_sq_assignment", Args, trip_tab, tod, mode)
                    end
                end
            
                for per in {"pk", "np"} do
                    lx_tag = mr.RunCode("get_pa_file_tag", "air", per, "lx")          
                    trip_tab = mr.RunCode("get_segment_mc_trip_file", Args.OutputFolder, lx_tag)

                    if per = "pk" then tod = "am" else tod = "md"
                    ok = mr.RunCode("transit_sq_assignment", Args, trip_tab, tod, "lx")
                end

                if ok = 1 then
                    ShowMessage("Transit Select Query Analysis finished.")
            end
        end
        enditem

EndMenu 
    // Add a Cancel button. The Cancel keyword allows the user to press Esc.



