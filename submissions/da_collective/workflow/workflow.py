
import os
import sys
import shutil
import platform
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import rel_entr,kl_div
from scipy.stats import wasserstein_distance as wd
from multiprocessing import Pool,Manager,Process
# use locally stored versions
sys.path.insert(0,os.path.join("dependencies"))
sys.path.insert(0,".")
import pyemu
import pastas as ps


def setup_pst(ws,use_phi_factors=True):
    """setup the pest interface
    """

    new_d = ws + "_template"
    input_df = get_input_df(ws)
    np.random.seed(pyemu.en.SEED)
    pf = pyemu.utils.PstFrom(ws,new_d=new_d,start_datetime=input_df.index[0],remove_existing=True)
    results = pd.read_csv(os.path.join(ws,"results.csv"),index_col=0)
    pf.add_observations("results.csv",index_cols="datetime",use_cols=list(results.columns),ofile_sep=",")
    #pf.add_observations("sigsum.csv", index_cols=["signame"], use_cols=["sigval"], ofile_sep=",")
    #pf.add_observations("predsum.csv", index_cols=["predname"], use_cols=["predval"], ofile_sep=",")

    # add additional observations to monitor sim gw levels WRT land surface
    if "netherlands" in ws:
        pf.add_observations("simdup.csv",index_cols=["datetime"],use_cols=["simdup"],ofile_sep=",")

    # add the smoothed sim gw levels for trends
    smooth = setup_smoothed_obs(pf.new_d)
    pf.add_observations("smoothed.csv",index_cols=["datetime"],use_cols=smooth.columns.tolist(),ofile_sep=",",
                        prefix="smooth",obsgp="smooth")

    # add pastas parameters
    pastas_pars = pd.read_csv(os.path.join(ws,"pastas_pars.csv"),index_col=0)
    pastas_pars_df = pf.add_parameters("pastas_pars.csv",par_type="grid",par_style="direct",
                      index_cols=["pastaname"],use_cols=["initial"],
                      pargp="pastas",par_name_base="pastas",mfile_sep=",",transform="none")

    # add forcing pars - one set of daily pars and one set of julian day pars
    use_cols = ["tmean","precip","pet"]
    if "stage" in input_df.columns:
        use_cols.append("stage")
    ub,lb = 1.9,0.1
    #if "netherlands" in new_d:
    #    ub,lb = 1.9,0.1
    pf.add_parameters("input_data.csv",par_type="grid",index_cols=["datetime"],
                      use_cols=use_cols,pargp=use_cols,par_name_base=use_cols,lower_bound=[lb,lb,lb],
                      upper_bound = [ub,ub,ub],ult_lbound=[-1.0e+30,0.0,0.0],transform="none")
    setup_julian_day_input_pars(pf.new_d,use_cols)
    pf.add_py_function("workflow.py","apply_julian_day_input_pars()",is_pre_cmd=True)
    pf.add_parameters("julian_day_pars.csv",par_type="grid",index_cols=["julian-day"],use_cols=use_cols,upper_bound=5,lower_bound=0.2,
                      transform="log",pargp=["julian-"+u for u in use_cols],par_name_base=["julian-"+u for u in use_cols],
                      par_style="direct")

    # extra py bits
    pf.extra_py_imports.append("pandas as pd")
    pf.extra_py_imports.append("pastas as ps")
    pf.extra_py_imports.append("numpy as np")

    pf.add_py_function("workflow.py","get_input_df(ws='.')",is_pre_cmd=None,)
    pf.add_py_function("workflow.py", "get_output_df(ws='.')", is_pre_cmd=None)
    pf.add_py_function("workflow.py", "run_pastas_model()", is_pre_cmd=None)
    pf.add_py_function("workflow.py", "get_pastas_model(input_df,output_df)", is_pre_cmd=None)
    pf.add_py_function("workflow.py","smooth(df)",is_pre_cmd=None)
    pf.add_py_function("workflow.py","process_smoothed_obs()",is_pre_cmd=False)

    pf.mod_py_cmds.append("run_pastas_model()")
    
    # build the control file
    pst = pf.build_pst()


    # set some better prior parameter estimates
    par = pst.parameter_data
    ppar = par.loc[par.pname=='pastas',:].copy()
    output_df = get_output_df(pf.new_d)
    for parnme,pastaname in zip(ppar.index,ppar.pastaname.values):
        transform = "none"
        #if pastas_pars.loc[pastaname,"vary"] == False:
        #    transform = "fixed"
        par.loc[parnme, "partrans"] = transform
        par.loc[parnme,"pargp"] = pastaname
        par.loc[parnme,"standard_deviation"] = np.abs(par.loc[parnme,"parval1"] * 0.75)
        if "constant" in pastaname:
            par.loc[parnme, "parubnd"] = par.loc[parnme, "parval1"] + 3#?
            par.loc[parnme, "parlbnd"] = par.loc[parnme, "parval1"] - 3#?
            par.loc[parnme, "standard_deviation"] = 0.25
        elif "stage" in pastaname:
            par.loc[parnme, "parubnd"] = par.loc[parnme, "parval1"] + 5  # ?
            par.loc[parnme, "parlbnd"] = par.loc[parnme, "parval1"] - 5  # ?
            par.loc[parnme, "standard_deviation"] = 0.25
        elif "upperthres-1" in pastaname:
            par.loc[parnme, "parlbnd"] = output_df.loc[:,"head"].min() - output_df.loc[:,"head"].std()
            par.loc[parnme, "parubnd"] = output_df.loc[:,"head"].max() + output_df.loc[:,"head"].std()
            par.loc[parnme, "standard_deviation"] = 0.3
        else:
            par.loc[parnme,"parubnd"] = pastas_pars.loc[pastaname,"pmax"]
            par.loc[parnme, "parlbnd"] = pastas_pars.loc[pastaname, "pmin"]

            #par.loc[parnme, "parubnd"] = par.loc[parnme, "parval1"] + np.abs(par.loc[parnme, "parval1"] * 0.5)
            #par.loc[parnme, "parlbnd"] = par.loc[parnme, "parval1"] - np.abs(par.loc[parnme, "parval1"] * 0.5)



    # write the control file and test run
    # pst.control_data.noptmax = 0
    # pst.write(os.path.join(pf.new_d,"pest.pst"),version=2)
    # pyemu.os_utils.run("pestpp-ies pest.pst",cwd=pf.new_d)
    # pst.set_res(os.path.join(pf.new_d,"pest.base.rei"))
    # print(pst.phi)
    # assert pst.phi < 0.5

    # now set the observed values
    # first the rolling/smoothed obs
    output_df = get_output_df(pf.new_d)
    if "sweden" in ws:
        dts = pd.date_range(output_df.index[0],output_df.index[-1],freq='d')
        interp_df = output_df.reindex(dts)
        interp_df.loc[:,"head"] = interp_df.loc[:,"head"].interpolate()
    else:
        interp_df = output_df
    obs = pst.observation_data
    obs.loc[:,"weight"] = 0.0
  
    robs = obs.loc[obs.obsnme.str.contains("roll"),:].copy()
    robs.loc[:,"datetime"] = pd.to_datetime(robs.datetime)
    obs.loc[robs.obsnme,"obgnme"] = robs.usecol
    usecols = robs.usecol.unique()
    usecols.sort()
    windows = [int(c.split('-')[1]) for c in usecols]
    for usecol,window in zip(usecols,windows):
        uobs = robs.loc[robs.usecol==usecol,:].copy()
        routput = interp_df.loc[:,"head"].rolling(window).mean()
        routput.dropna(inplace=True)
        uobs.index = uobs.datetime
        uobs = uobs.loc[routput.index,:].copy()
        uobs.loc[:,"obsval"] = routput.values
        naobs = uobs.loc[pd.isna(uobs.obsval),:]
        if naobs.shape[0] > 0:
            print(routput)
            routput.to_csv("routput.csv")
            naobs.to_csv("naobs.csv")
            print(naobs)
            raise Exception()
        obs.loc[uobs.obsnme,"obsval"] = uobs.obsval.values
        obs.loc[uobs.obsnme, "observed"] = True
        obs.loc[uobs.obsnme, "weight"] = 1.0
        obs.loc[uobs.obsnme, "standard_deviation"] = 0.01

    # the the raw gw sim values
    gobs = obs.loc[obs.obsnme.str.contains("gwsim"),:].copy()
    gobs.loc[:,"datetime"] = pd.to_datetime(gobs.datetime)
    gobs.index = gobs.datetime
    if gobs.index.has_duplicates:
        print(gobs.index.duplicated())
        raise Exception()
    gobs = gobs.loc[output_df.index,:]
    gobs.loc[:,"obsval"] = output_df.values
    obs.loc[gobs.obsnme.values,"obsval"] = gobs.obsval.values
    obs.loc[gobs.obsnme.values, "weight"] = 10.0
    # give more weight to more recent obs
    recent_gobs = gobs.loc[gobs.datetime > pd.to_datetime("1-1-14"),:]
    obs.loc[recent_gobs.obsnme.values,"weight"] *= 3
    print("weight values",obs.weight.unique())

    # set the standard deviation for noise estimation later
    obs.loc[gobs.obsnme.values, "standard_deviation"] = output_df.loc[:,"head"].std() * 0.2
    if "netherlands" in new_d:
        obs.loc[gobs.obsnme.values, "standard_deviation"] = output_df.loc[:,"head"].std() * 0.3
    
    obs.loc[gobs.obsnme.values, "observed"] = True
    # set a "distance" for drawing autocorrelated noise realizations
    obs.loc[gobs.obsnme.values,"distance"] = np.abs(gobs.datetime.rsub(gobs.datetime.min()).dt.days.values)

    # Set the observed value for the summary stats
    # obs_sigsum = ps.stats.signatures.summary(output_df.loc[:,"head"].copy())
    # obs_sigsum.index = [i.replace("_", "-") for i in obs_sigsum.index]
    # sobs = obs.loc[obs.usecol=="sigval",:].copy()
    # sobs.index = sobs.signame
    # #print(sobs.signame)
    # if sobs.index.has_duplicates:
    #     print(sobs.index.duplicated())
    #     raise Exception()
    # #sobs = sobs.loc[sobs.obsval < 1.e30,:]
    # sobs.loc[obs_sigsum.index,"obsval"] = obs_sigsum.values
    # obs.loc[sobs.obsnme.values,"obsval"] = sobs.obsval.values
    # obs.loc[sobs.obsnme.values, "observed"] = True
    # obs.loc[sobs.obsnme.values, "standard_deviation"] = sobs.obsval.values * 0.15
    # obs.loc[sobs.obsnme.values, "weight"] = 0.0

    # set the "perfect" value for the efficiency metrics
    # eff_obs = obs.loc[obs.obsnme.apply(lambda x: "nse" in x or "kge" in x),"obsnme"]
    # obs.loc[eff_obs,"obsval"] = 1.0
    # obs.loc[eff_obs,"obgnme"] = "eff_metrics"
    # obs.loc[eff_obs,"weight"] = 1.0
    # obs.loc[eff_obs, "standard_deviation"] = 0.00001

    # set some ineq obs for netherlands
    if "netherlands" in ws:
        simdup = obs.loc[obs.usecol=="simdup",:].copy()
        print(simdup)
        assert simdup.shape[0] > 0
        obs.loc[simdup.obsnme,"obsval"] = output_df.values.max()
        #obs.weight.loc[simdup.obsnme] = 1.0
        obs.loc[simdup.obsnme,"obgnme"] = "less_than_gwsim"
        print(obs.loc[pst.nnz_obs_names,"obgnme"].unique())


    # prep for auto correlated forcing parameter realizations
    jpar = par.loc[par.parnme.str.contains("julian"),:]
    jpar.loc[:,"x"] = jpar.loc[:,"julian-day"].astype(float)
    jpar.loc[:, "y"] = 0.0
    pnames = jpar.pname.unique()
    sd = {}
    for pname in pnames:
        v = pyemu.geostats.ExpVario(contribution=1.0, a=150) # days
        gs = pyemu.geostats.GeoStruct(variograms=v,name=pname)
        ppar = jpar.loc[jpar.pname==pname,:].copy()
        ppar.sort_values(by="julian-day",inplace=True)
        sd[gs] = ppar

    dpar = par.loc[par.usecol.isin(use_cols),:].copy()
    dpar = dpar.loc[par.parnme.apply(lambda x: "julian" not in x),:]
    dpar.loc[:,"datetime"] = pd.to_datetime(dpar.datetime)
    for use_col in use_cols:
        v = pyemu.geostats.ExpVario(contribution=1.0, a=90) # days
        gs = pyemu.geostats.GeoStruct(variograms=v, name=use_col+"daily")
        ddpar = dpar.loc[dpar.usecol == use_col,:].copy()
        ddpar.sort_values(by="datetime",inplace=True)
        ddpar.loc[:,"x"] = (ddpar.datetime - ddpar.datetime.min()).dt.days
        ddpar.loc[:,"y"] = 1
        sd[gs] = ddpar
    
    # draw
    pe = pyemu.helpers.geostatistical_draws(pst,struct_dict=sd,num_reals=1000)#pf.draw(num_reals=1000)
    pe.enforce()
    pe.to_binary(os.path.join(pf.new_d,"prior.jcb"))
    pst.pestpp_options["ies_par_en"] = "prior.jcb"

    # now fix the pred dpars - we want these to be stochastic but not estimated
    dpar = par.loc[par.usecol.isin(use_cols), :].copy()
    dpar.loc[:,"datetime"] = pd.to_datetime(dpar.datetime)
    dpar = dpar.loc[par.parnme.apply(lambda x: "julian" not in x), :]
    dpar = dpar.loc[dpar.datetime>output_df.index[-1],:]
    par.loc[dpar.parnme,"partrans"] = "fixed"


    # now for autocorrelated noise realizations
    v = pyemu.geostats.ExpVario(contribution=1.0,a=120) # days
    gs = pyemu.geostats.GeoStruct(variograms=v)
    sd = {gs: gobs.obsnme.tolist()}
    noise = pyemu.helpers.autocorrelated_draw(pst,struct_dict=sd,num_reals=1000)
    noise.to_binary(os.path.join(pf.new_d,"noise.jcb"))
    pst.pestpp_options["ies_obs_en"] = "noise.jcb"

    #now that the noise is drawn, regroup the gwsim obs into quantiles for weighting purposes
    gobs = obs.loc[gobs.obsnme.values,:]
    pvals = [0,5,25,50,75,95,100]
    vals = np.percentile(gobs.obsval.values,pvals)
    obs.loc[gobs.obsnme,"obgnme"] = "gwsim-p000"
    for pval,val in zip(pvals,vals):
        pobsnme = gobs.loc[gobs.obsval > val,"obsnme"]
        print(pval,val,pobsnme.shape[0])
        obs.loc[pobsnme.values,"obgnme"] = "gwsim-p{0:03.0f}".format(pval)
    print(obs.obgnme.unique())
    print(obs.obgnme.value_counts())

    #obs.loc[sobs.obsnme.values, "weight"] = 1.0
    #obs.loc[sobs.obsnme.values, "standard_deviation"] = sobs.obsval * 0.1 #?

    # some basic pestpp-ies options
    pst.pestpp_options["ies_num_reals"] = 100
    pst.pestpp_options["ies_subset_size"] = -10
    pst.pestpp_options["ies_bad_phi_sigma"] = 1.75
    pst.pestpp_options["par_sigma_range"] = 10

    pst.pestpp_options["ies_save_binary"] = True

    # use some site specific weighting/likelihood function stratigies
    if use_phi_factors:
        grps = pst.nnz_obs_groups
        df = pd.DataFrame({"fac":np.ones(len(grps))},index=grps)
        df.loc[:,"fac"] = 1.0/len(grps)
        if "nether" in ws:
            df.loc["gwsim-p000", "fac"] *= 1.1
            df.loc["gwsim-p005", "fac"] *= 1.1
            df.loc["gwsim-p075", "fac"] *= 1.2
            
        elif "germany" in ws:
            df.loc["gwsim-p000", "fac"] *= 2.0
            df.loc["gwsim-p005", "fac"] *= 2.0
        # elif "sweden1" in ws:
        #     df.loc["gwsim-p075", "fac"] *= 1.1
        #     df.loc["gwsim-p095", "fac"] *= 1.5
        #     df.loc["gwsim-p005", "fac"] *= 1.25

        else:
            df.loc["gwsim-p000","fac"] *= 1.5
            df.loc["gwsim-p005","fac"] *= 1.25
            df.loc["gwsim-p095","fac"] *= 1.5
        #with open(os.path.join(pf.new_d,"phi.csv"),'w') as f:
        #    for grp in grps:
        #        f.write("{0},{1}\n".format(grp,1.0/(len(grps))))
        df.to_csv(os.path.join(pf.new_d,"phi.csv"),header=False)
        pst.pestpp_options["ies_phi_factor_file"] = "phi.csv"

    # test run
    pst.write(os.path.join(pf.new_d, "pest.pst"), version=2)
    pyemu.os_utils.run("pestpp-ies pest.pst", cwd=pf.new_d)

    # test run
    pst.control_data.noptmax = -2
    pst.write(os.path.join(pf.new_d, "pest.pst"), version=2)
    pyemu.os_utils.run("pestpp-ies pest.pst", cwd=pf.new_d)

    pst.control_data.noptmax = 4
    pst.write(os.path.join(pf.new_d, "pest.pst"), version=2)

def run(t_d,m_d=None,noptmax=None,num_workers=20,**kwargs):
    """run pestpp-ies in parallel locally
    """
    port = 4201
    worker_t_d = t_d+"_clean"
    prep_worker(t_d,worker_t_d)
    pst = pyemu.Pst(os.path.join(t_d,"pest.pst"))
    if noptmax is not None:
        pst.control_data.noptmax = noptmax
    for k,v in kwargs.items():
        pst.pestpp_options[k] = v
    if m_d is None:
        m_d = t_d.replace("template","master")

    pst.write(os.path.join(t_d,"pest.pst"),version=2)

    #pyemu.os_utils.start_workers(t_d,"pestpp-ies","pest.pst",num_workers=num_workers,master_dir=m_d)
    if os.path.exists(m_d):
        shutil.rmtree(m_d,ignore_errors=True)
    shutil.copytree(t_d,m_d)
    cwd = os.getcwd()
    os.chdir(m_d)
    if sys.platform.startswith('linux') or sys.platform.startswith("dar") or sys.platform.startswith("mac"):
        args = ["./pestpp-ies","pest.pst","/h",":{0}".format(port)]
    else: 
        args = ["pestpp-ies.exe","pest.pst","/h",":{0}".format(port)]
    master_p = subprocess.Popen(args)
    os.chdir(cwd)

    pyemu.os_utils.start_workers(t_d,"pestpp-ies","pest.pst",num_workers=num_workers,worker_root=os.path.join("models"),
                                 port=port)
    master_p.wait()

def prep_worker(org_d, new_d):
    """prepare a clean working dir to save storage
    """
    if os.path.exists(new_d):
        shutil.rmtree(new_d,ignore_errors=True)
    shutil.copytree(org_d,new_d)
    exts = ["jcb","rei","dbf","shp","shx"]

    files = [f for f in os.listdir(new_d) if f.lower().split('.')[-1] in exts]
    for f in files:
        os.remove(os.path.join(new_d,f))
    mlt_dir = os.path.join(new_d,"mult")
    for f in os.listdir(mlt_dir)[1:]:
        os.remove(os.path.join(mlt_dir,f))
    



def get_pastas_model(input_df,output_df):
    """get a basic pastas model
    """
    
    ml = ps.Model(output_df.loc[:,"head"].copy(),noisemodel=False)
    sm = ps.RechargeModel(input_df.precip.copy(), input_df.pet.copy(), recharge=ps.rch.FlexModel(snow=True),
                          rfunc=ps.FourParam, name="recharge", temp=input_df.tmean.copy())
    ml.add_stressmodel(sm)
    ml.add_transform(ps.ThresholdTransform(value=output_df.loc[:,"head"].quantile(0.8),name="upperthres"))
    if "stage" in input_df.columns:
        smstage = ps.StressModel(input_df.stage.copy(),ps.One,name="stage",settings="waterlevel")
        ml.add_stressmodel(smstage)
    return ml

def test_pastas_model(t_d):
    """prepare and run (and calibrate) a basic pastas model for a given 
    template directory
    """

    input_df = get_input_df(t_d)
    input_df.index.name = "datetime"
    input_df.to_csv(os.path.join(t_d,"input_data.csv"))
    output_df = get_output_df(t_d)
    ml = get_pastas_model(input_df,output_df)
    ml.simulate()
    pardf = ml.get_init_parameters()

    # for whatever reason, some of the site fail to solve with 
    # the threshold transform...
    try:
        ml.solve()
    except Exception as e:
        print("failed to solve, removing transform")
        ml = get_pastas_model(input_df, output_df)
        ml.del_transform()
        ml.simulate()
        ml.solve()

    # get the optimal par values
    pardfopt = ml.get_init_parameters(initial=False)
    pardf.loc[pardfopt.index,:] = pardfopt.loc[:,:].values

    # if not netherlands, expland the bounds
    if "netherlands" not in t_d:
        pardf.loc["recharge_srmax", "pmax"] *= 20
        pardf.loc["recharge_lp", "pmin"] /= 10
        pardf.loc["recharge_kv", "pmin"] /= 10
        pardf.loc["recharge_kv", "pmax"] *= 10
        pardf.loc["upperthres_2", "pmax"] *= 10
        pardf.loc["recharge_b", "pmax"] *= 3
        pardf.loc["recharge_b", "pmin"] /= 100

    # some minor renaming to help later
    ivals = [i.replace("_","-").lower() for i in pardf.index.values]
    ivals[0] = "recharge-biga"
    pardf.index = ivals
    pardf.index.name = "pastaname"
    pardf.loc[:,"optimal"] = np.nan
    pardf.to_csv(os.path.join(t_d,"pastas_pars.csv"))


    # check that the pastas model runs using an external par file
    b_d = os.getcwd()
    os.chdir(t_d)
    run_pastas_model()
    os.chdir(b_d)
    print(pardf)
    return ml

def run_pastas_model():
    """run the pastas model using an external par file.  This fxn is called at
    runtime
    """
    import pastas as ps
    input_df = get_input_df(".")
    output_df = get_output_df(".")

    ml = get_pastas_model(input_df,output_df)
    pardf = pd.read_csv("pastas_pars.csv",index_col=0)
    ml.parameters = pardf
    simdf = ml.simulate()
    rchdf = ml.stressmodels["recharge"].get_water_balance(ml.get_parameters("recharge"))
    rchdf.columns = [c.lower().replace(" ","-").replace("(","").replace(")","") for c in rchdf.columns]
    rchdf.loc[:,"gwsim"] = simdf
    rchdf.index.name = "datetime"
    rchdf.to_csv("results.csv")
    gwsim = rchdf.loc[:,["gwsim"]]
    gwsim.loc[:,"simdup"] = gwsim.pop("gwsim")
    gwsim.to_csv("simdup.csv")
    #preddf = simdf.loc[simdf.index > output_df.index[-1]]
    #simdf = simdf.loc[output_df.index]
    #sigsum = pd.DataFrame({"sigval":ps.stats.signatures.summary(simdf)})
    #sigsum.index = [i.replace("_","-") for i in sigsum.index]
    #sigsum.index.name = "signame"
    #sigsum = sigsum.fillna(1e+30)
    # sigsum = pd.DataFrame(columns=["sigval"])
    # sigsum.index.name="signame"
    # sigsum.loc["nse","sigval"] = ml.stats.nse()
    # sigsum.loc["kge", "sigval"] = ml.stats.kge_2012()

    # sigsum.to_csv("sigsum.csv")

    # sigsum = pd.DataFrame({"predval": ps.stats.signatures.summary(preddf)})
    # sigsum.index = [i.replace("_", "-") for i in sigsum.index]
    # sigsum.index.name = "predname"
    # sigsum = sigsum.fillna(1e+30)
    # sigsum.to_csv("predsum.csv")

    return rchdf


def prep_dir(org_d,ws):
    if os.path.exists(ws):
        shutil.rmtree(ws)
    os.makedirs(ws)
    for f in os.listdir(org_d):
        shutil.copy2(os.path.join(org_d,f),os.path.join(ws,f))
    prep_deps(ws)



def get_input_df(ws='.'):
    """get a consistent input dataframe for pastas
    """
    name_dict = {"rr":"precip","tg":"tmean","tn":"tmin","tx":"tmax","pp":"pressure",\
                "hu":"humidity","fg":"wind","qq":"rad","et":"pet","prcp":"precip","stage_m":"stage","et":"pet"}
    df = pd.read_csv(os.path.join(ws,"input_data.csv"),index_col=0,parse_dates=[0])
    df.columns = [c.lower() for c in df.columns]
    cols = set(df.columns.tolist())
    for old,new in name_dict.items():
        if old in cols:
            df.loc[:,new] = df.pop(old)
    if "tmean" not in df.columns:
        df.loc[:,"tmean"] = (df.tmin + df.tmax) / 2.0
    return df

def get_output_df(ws='.'):
    """get a consistent output dataframe for pastas
    """
    df = pd.read_csv(os.path.join(ws,"heads.csv"),index_col=0,parse_dates=[0])
    return df

def prep_deps(d):
    """prep dependencies within a given directory
    """
    dd = "dependencies"
    for org_dd in [_d for _d in os.listdir(dd) if os.path.isdir(os.path.join(dd,_d))]:
        shutil.copytree(os.path.join(dd,org_dd),os.path.join(d,org_dd))
    if "macos" in platform.platform().lower():
        org_d = os.path.join("bin","mac")
    elif "win" in platform.platform().lower():
        org_d = os.path.join("bin","win")
    else:
        org_d = os.path.join("bin", "linux")
    for b in os.listdir(org_d):
        shutil.copy2(os.path.join(org_d,b),os.path.join(d,b))



def plot_results(m_d,keep_best=-0.9,post_iter=None,percentile_vals=[5,95],plt_name=None):
    """just some ugly figs for diagnostics
    """
    start = datetime.now()
    print("-->plotting", m_d)
    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    pst.try_parse_name_metadata()
    obs = pst.observation_data
    obs.loc[:,"datetime"] = pd.to_datetime(obs.datetime)
    obs_usecols = obs.usecol.unique()
    obs_usecols.sort()
    par = pst.parameter_data
    #par.loc[:,"datetime"] = pd.to_datetime(par.datetime)
    output_df = get_output_df(m_d)
    start_dt = output_df.index[0]
    noise = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(m_d,"pest.obs+noise.jcb"))
    pe_dict = {}
    oe_dict = {}
    iters = []

    for i in range(pst.control_data.noptmax+1):
        pe_file = os.path.join(m_d,"pest.{0}.par.jcb".format(i))
        oe_file = os.path.join(m_d, "pest.{0}.obs.jcb".format(i))

        if os.path.exists(pe_file) and os.path.join(oe_file):
            iters.append(i)

    if len(iters) > 1:
        if post_iter is not None:
            if post_iter in iters:
                iters = [post_iter]
            else:
                iters = iters[-1:]
    if 0 not in iters:
        iters.insert(0,0)

    for i in iters:
        pe_file = os.path.join(m_d, "pest.{0}.par.jcb".format(i))
        oe_file = os.path.join(m_d, "pest.{0}.obs.jcb".format(i))
        oe = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=oe_file)
        pe = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=pe_file)
        if keep_best < 0:
            kkeep_best = int(oe.shape[0] * np.abs(keep_best))
        else:
            kkeep_best = keep_best
        if oe.shape[0] > kkeep_best:
            pv = oe.phi_vector
            pv.sort_values(inplace=True,ascending=True)
            keep = pv.iloc[:kkeep_best].index.values
            oe = oe.loc[keep,:]
            pe = pe.loc[keep,:]

        pe_dict[i] = pe
        oe_dict[i] = oe

    pr_pe = pe_dict.pop(0)
    pr_oe = oe_dict.pop(0)
    #iters = [iters[-1]]

    jpar = par.loc[par.pname.str.contains("julian"), :].copy()
    jpar.loc[:, "julian-day"] = jpar.loc[:, "julian-day"].astype(float)
    jtypes = jpar.pname.unique()
    jtypes.sort()

    pastapars = par.loc[par.parnme.str.contains("pastas"), :]

    daily_pnames = ["tmean", "pet", "precip"]
    if "stage" in par.pname.unique():
        daily_pnames.append('stage')
    dpars = par.loc[par.pname.isin(daily_pnames), :].copy()
    dpars.loc[:, "datetime"] = pd.to_datetime(dpars.datetime)

    phidf = pd.read_csv(os.path.join(m_d,"pest.phi.actual.csv"),index_col=0)
    org_plt_name = plt_name
    

    for i in iters:
        if i ==0:
            continue
        print("...plotting iteration",i,"for ",m_d)
        if org_plt_name is None:
            plt_name = os.path.join(m_d, "results_{0}_{1}.pdf".format(i,os.path.split(m_d)[-1]))
        else:
            plt_name = org_plt_name
        with PdfPages(plt_name) as pdf:

            for pname in pastapars.parnme:
                fig,ax = plt.subplots(1,1,figsize=(6,6))
                ax.hist(pr_pe.loc[:,pname].values,bins=20,facecolor="0.5",alpha=0.5,density=True)
                if i > 0:
                    ax.hist(pe_dict[i].loc[:, pname].values, bins=20, facecolor="b", alpha=0.5,density=True)
                ylim = ax.get_ylim()
                lb = pastapars.loc[pname,"parlbnd"]
                ub = pastapars.loc[pname, "parubnd"]
                xlim = ax.get_xlim()
                ax.plot([lb,lb],ylim,"r-")
                ax.plot([ub, ub], ylim, "r-")
                ax.set_xlim(xlim)
                ax.set_title(pastapars.loc[pname,"pastaname"],loc="left")
                ax.set_yticks([])
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)

            for jtype in jtypes:
                jjpar = jpar.loc[jpar.pname==jtype,:].copy()
                jjpar.sort_values(by="julian-day",inplace=True)
                fig,ax = plt.subplots(1,1,figsize=(15,5))
                pr_vals = pr_pe.loc[:,jjpar.parnme].values
                jvals = jjpar.loc[:,"julian-day"].values
                for ireal in range(pr_vals.shape[0]):
                    ax.plot(jvals,pr_vals[ireal,:],"0.5",lw=0.1,alpha=0.5)
                if i > 0:
                    pt_vals = pe_dict[i].loc[:,jjpar.parnme].values
                    for ireal in range(pt_vals.shape[0]):
                        ax.plot(jvals, pt_vals[ireal, :], "b", lw=0.1, alpha=0.5)
                xlim = ax.get_xlim()
                ax.plot(xlim,[1,1],"k--")
                ax.set_title(jtype,loc="left")
                ax.set_xlabel("julian day")
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)



            for pname in daily_pnames:
                ppars = dpars.loc[dpars.pname==pname,:].copy()
                ppars.sort_values(by="datetime",inplace=True)
                ppars = ppars.loc[ppars.datetime >= start_dt,:]
                dt_vals = ppars.datetime.values
                pr_vals = pr_pe.loc[:,ppars.parnme].values
                fig,ax = plt.subplots(1,1,figsize=(15,5))
                for ireal in range(pr_vals.shape[0]):
                    ax.plot(dt_vals,pr_vals[ireal,:],"0.5",lw=0.1,alpha=0.5)
                if i > 0:
                    pt_vals = pe_dict[i].loc[:,ppars.parnme].values
                    for ireal in range(pt_vals.shape[0]):
                        ax.plot(dt_vals,pt_vals[ireal,:],"b-",lw=0.1,alpha=0.5)
                ax.set_title(pname,loc="left")

                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)


            # fig,ax = plt.subplots(1,1,figsize=(10,5))
            # [ax.plot(phidf.index.values,phidf.iloc[:,ireal].values,"0.5",alpha=0.5) for ireal in range(6,phidf.shape[1])]
            # ax.set_title("phi",loc="left")
            # plt.tight_layout()
            # pdf.savefig()
            # plt.close(fig)

            for usecol in obs_usecols:
                if usecol != "gwsim":# and usecol != "sigval" and usecol.split('-')[0] != "roll":
                     continue

                uobs = obs.loc[obs.usecol==usecol,:].copy()
                uobs.sort_values(by="datetime",inplace=True)
                if pd.notna(uobs.datetime.iloc[0]):
                    uobs = uobs.loc[uobs.datetime >= start_dt, :]
                pr_vals = pr_oe.loc[:, uobs.obsnme].values
                pr_base_vals = None
                if "base" in pr_oe.index:
                    pr_base_vals = pr_oe.loc["base",uobs.obsnme].values
                pt_vals,pt_base_vals = None,None
                if i > 0:
                    pt_vals = oe_dict[i].loc[:, uobs.obsnme].values
                    if "base" in oe_dict[i].index:
                        pt_base_vals =oe_dict[i].loc["base", uobs.obsnme].values

                if pd.notna(uobs.datetime.iloc[0]):
                    fig,axes = plt.subplots(3,1,figsize=(20,15))

                    full_dt_vals = uobs.datetime.values
                    uobs = uobs.loc[uobs.observed == True,:]
                    dt_vals = uobs.datetime.values
                    for ax in axes:
                        ax.scatter(dt_vals,uobs.obsval,marker="^",c="r",s=5,label="observed",zorder=10)
                    ylim = axes[0].get_ylim()
                    if uobs.shape[0] > 0:
                        noise_vals = noise.loc[pr_oe.index,uobs.obsnme].values
                        for ireal in range(noise_vals.shape[0]):
                            label = None
                            if ireal == 0:
                                label = "noise"
                            for ax in axes[[1]]:
                                ax.plot(dt_vals,noise_vals[ireal,:],"r-",lw=0.1,alpha=0.25,label=label)

                    if pt_vals is not None:
                        for ireal in range(pt_vals.shape[0]):
                            label = None
                            if ireal == 0:
                                label = "posterior"
                            for ax in axes[[1]]:
                                ax.plot(full_dt_vals, pt_vals[ireal, :], "b-", lw=0.2, alpha=0.25,label=label)
                        if pt_base_vals is not None:
                            for ax in axes[[0]]:
                                ax.plot(full_dt_vals,pt_base_vals,"b-",lw=2.5,label="posterior base",alpha=0.5)
                                ax.plot(full_dt_vals,pt_vals.mean(axis=0),"m-",lw=2.5,label="posterior mean",alpha=0.5)

                        cl1ub,cl1lb = [],[]
                        cl2ub, cl2lb = [], []
                        for ival in range(pt_vals.shape[1]):
                            llb,uub = np.percentile(pt_vals[:,ival],percentile_vals)
                            cl1ub.append(uub)
                            cl1lb.append(llb)
                            # llb, uub = np.percentile(pt_vals[:, ival], [10, 90])
                            # cl2ub.append(uub)
                            # cl2lb.append(llb)
                        ub = pt_vals.max(axis=0)
                        lb = pt_vals.min(axis=0)
                        #axes[2].fill_between(full_dt_vals,lb,ub,facecolor="b",alpha=0.5,label="minmax")
                        #axes[2].fill_between(full_dt_vals, cl2lb, cl2ub, facecolor="g", alpha=0.85)
                        axes[2].fill_between(full_dt_vals, cl1lb, cl1ub, facecolor="b", alpha=0.5)

                        if uobs.shape[0] > 0:
                            axt = plt.twinx(axes[1])
                            noise_vals = noise.loc[oe_dict[i].index,uobs.obsnme].values
                            pt_vals = oe_dict[i].loc[:, uobs.obsnme].values
                            #kl = [sum(rel_entr(noise_vals[:,ival],pt_vals[:,ival])) for ival in range(uobs.shape[0])]
                            #kl = np.array([kl_div(noise_vals[:,ival],pt_vals[:,ival]) for ival in range(uobs.shape[0])])
                            #axt.plot(dt_vals,kl,"0.5",lw=1.0)
                            #axt.set_ylabel("KL divergence",color="0.5")
                            #axt.set_ylim(kl.max(),0)
                            zwobs_idx = uobs.weight == 0
                            print(zwobs_idx)
                            kl = np.array([wd(pt_vals[:,ival],noise_vals[:,ival]) for ival in range(uobs.shape[0])])
                            kl[zwobs_idx.values] = np.nan
                            #axt.plot(dt_vals,kl,"0.5",lw=1.0)
                            axt.fill_between(dt_vals,0,kl,facecolor="0.5",alpha=0.6,interpolate=True)
                            axt.set_ylabel("earth mover distance",color="0.5")
                            axt.set_ylim(np.nanmax(kl)*2,0)

                            axt2 = plt.twinx(axes[2])
                            pt_mn = pt_vals.mean(axis=0)
                            pt_sd = pt_vals.std(axis=0)
                            stat_dist = ((uobs.obsval.values - pt_mn)**2)/pt_sd
                            axt2.fill_between(dt_vals,0,stat_dist,facecolor="0.5",alpha=0.7)
                            axt2.set_ylabel('statistical distance ($\\frac{obs - \\mu}{\\sigma}$)')
                            axt2.set_ylim(stat_dist.max()*2,0)

                    #    ax.set_ylim(uobs.obsval.min()-np.abs(uobs.obsval.std()*0.1),uobs.obsval.max()+np.abs(uobs.obsval.std()*0.1))
                    #if uobs.shape[0] == 0:
                    #ylim = axes[1].get_ylim()
                    if "roll" in usecol:
                        for ax in axes:
                            ax.plot(output_df.index,output_df.loc[:,"head"],"m-",lw=0.1,label="raw observed")
                        ymn = output_df.loc[:,"head"].min() - output_df.loc[:,"head"].std()*0.5
                        ymx = output_df.loc[:, "head"].max() + output_df.loc[:,"head"].std()*0.5
                        ylim = (ymn,ymx)

                    # for ireal in range(pr_vals.shape[0]):
                    #     label = None
                    #     if ireal == 0:
                    #         label = "prior"
                    #     axes[2].plot(full_dt_vals,pr_vals[ireal,:],"0.5",lw=0.1,alpha=0.25,zorder=0,label=label)

                    if pr_base_vals is not None:
                        for ax in axes[:-1]:
                            ax.plot(full_dt_vals,pr_base_vals,"k-",lw=1.0,dashes=(1,1),label="pastas best fit")

                    if "gwsim" in usecol and "netherlands" in m_d:
                        for ax in axes:
                            xlim = ax.get_xlim()
                            ax.plot(xlim,[11.35,11.35],"k--",lw=2,label="land surface")

                    for ax in axes:
                        ax.set_ylim(ylim)
                        ax.set_title(usecol,loc="left")
                        ax.legend(loc="upper left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)

                elif usecol == "sigval":
                    usigname = uobs.signame.unique()
                    usigname.sort()
                    for oname in uobs.obsnme:
                        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                        pr_vals = pr_oe.loc[:,oname].values
                        pr_vals[pr_vals>1e+20] = np.nan
                        ax.hist(pr_vals,bins=20,facecolor="0.5",alpha=0.5,density=True)
                        if i > 0:
                            pt_vals = oe_dict[i].loc[:, oname].values
                            pt_vals[pt_vals > 1e+20] = np.nan
                            ax.hist(pt_vals, bins=20, facecolor="b", alpha=0.5,density=True)
                        ax.hist(noise.loc[pr_oe.index,oname].values,facecolor="r",alpha=0.5,density=True)
                        oval = obs.loc[oname,"obsval"]
                        ylim = ax.get_ylim()
                        ax.plot([oval,oval],ylim,"r-",lw=3)
                        ax.set_title(obs.loc[oname,"signame"],loc="left")
                        ax.set_yticks([])
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close(fig)
    end = datetime.now()
    print('-->finished plotting ',m_d,", took:",(end-start).total_seconds(),' second')


def setup_julian_day_input_pars(t_d,use_cols):
    """prep the julian day forcing multipliers
    """

    #write generic julian day multipliers
    day = np.arange(1,367)
    df = pd.DataFrame({u:np.ones(day.shape[0]) for u in use_cols},index=day)
    df.index.name = "julian-day"
    df.to_csv(os.path.join(t_d,"julian_day_pars.csv"))

    b_d = os.getcwd()
    os.chdir(t_d)
    apply_julian_day_input_pars()
    os.chdir(b_d)

def apply_julian_day_input_pars():
    """apply the julian day forcing multipliers at runtime
    """
    df = pd.read_csv("julian_day_pars.csv",index_col=0)
    jd_nested_dict = {}
    for col in df.columns:
        jd_dict = df.loc[:,col].to_dict()
        jd_nested_dict[col] = jd_dict
    input_df = get_input_df(".")
    input_df.to_csv("org_input_data.csv")
    input_julian_day = input_df.index.dayofyear
    for col,jd_dict in jd_nested_dict.items():
        input_df.loc[:,col] *= np.array([jd_dict[day] for day in input_julian_day])
    input_df.to_csv("input_data.csv")

def prep_ned():
    """prep the netherlands interface
    """
    org_d = os.path.join("org_challenge_files","Netherlands")
    ws = os.path.join("models","netherlands")
    prep_dir(org_d,ws)

    ml = test_pastas_model(ws)
    setup_pst(ws)
    # input_df = get_input_df(ws)
    # input_df.index.name = "datetime"
    # input_df.to_csv(os.path.join(ws, "input_data.csv"))
    # output_df = get_output_df(ws)
    # ml = get_pastas_model(input_df, output_df)
    # ml.simulate()
    # ml.solve()
    #ml.plot()
    #plt.show()

def prep_usa():
    """prep the USA interface
    """
    org_d = os.path.join("org_challenge_files","USA")
    ws = os.path.join("models","usa")
    prep_dir(org_d,ws)

    ml = test_pastas_model(ws)
    setup_pst(ws)
    # input_df = get_input_df(ws)
    # input_df.index.name = "datetime"
    # input_df.to_csv(os.path.join(ws, "input_data.csv"))
    # output_df = get_output_df(ws)
    # ml = get_pastas_model(input_df, output_df)
    # ml.simulate()
    # ml.solve()
    # ml.plot()
    # plt.show()

def prep_germany():
    """prep the Germany interface
    """
    org_d = os.path.join("org_challenge_files", "Germany")
    ws = os.path.join("models", "germany")
    prep_dir(org_d, ws)
    ml = test_pastas_model(ws)
    setup_pst(ws)
    #ml.plot()
    #plt.show()

def prep_sweden1():
    """prep sweden1 interface
    """
    org_d = os.path.join("org_challenge_files", "Sweden_1")
    ws = os.path.join("models", "sweden1")
    prep_dir(org_d, ws)
    output_df = get_output_df(ws)
    start = pd.to_datetime("7-31-2002")
    end = pd.to_datetime("4-13-2003")
    output_df = output_df.loc[output_df.index.map(lambda x: x > end or x < start),:]
    output_df.to_csv(os.path.join(ws,"heads.csv"))

    test_pastas_model(ws)
    setup_pst(ws)
    # input_df = get_input_df(ws)
    # input_df.index.name = "datetime"
    # input_df.to_csv(os.path.join(ws, "input_data.csv"))
    # output_df = get_output_df(ws)
    # ml = get_pastas_model(input_df, output_df)
    # ml.simulate()
    # ml.solve()

    # ml.plot()
    # plt.show()


def prep_sweden2():
    """prep the sweden2 interace with some shananigans
    """
    org_d = os.path.join("org_challenge_files","Sweden_2")
    ws = os.path.join("models","sweden2")
    prep_dir(org_d,ws)
    # some cleaning of appearent BS in the data - those flat spots during low times
    # infill these with a cubic polynomial - just seems like the model needs to see
    # some lower water levels
    output_df = get_output_df(ws)
    output_df.loc[:,"idx"] = np.arange(output_df.shape[0])
    vc = output_df.loc[:,"head"].value_counts()
    # the problems appear less than elev 347 and are consecutive identical numbers
    vc = vc.loc[vc.index.map(lambda x: x < 347)]
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.scatter(output_df.index,output_df.loc[:,"head"].values,marker="^",c='0.5',s=8,label="org")
    ax.plot(output_df.index, output_df.loc[:, "head"].values,'0.5')
    fill_dts = []
    for v,c in vc.items():
        if c == 6:
            break
        cdf = output_df.loc[output_df.values==v].copy()
        # check for consecutives
        #cdf.loc[cdf.index[:-1],"idif"] = cdf.idx.values[:-1] - cdf.idx.values[1:]
        #cdf = cdf.loc[cdf.idif==-1]
        output_df.loc[cdf.index,"head"] = np.nan
        fill_dts.extend(cdf.index.tolist())

    ax.scatter(output_df.index, output_df.loc[:, "head"].values, marker="^", c='b', s=8,label="clean")

    dts = pd.date_range(output_df.index[0],output_df.index[-1],freq='d')
    routput = output_df.reindex(dts)
    iroutput = routput.interpolate(method="polynomial",order=3)
    ax.plot(iroutput.index,iroutput.loc[:,"head"].values,"m--")
    foutput_df = output_df.copy()
    foutput_df.loc[fill_dts,"head"] = iroutput.loc[fill_dts,"head"].values
    ax.scatter(foutput_df.index, foutput_df.loc[:, "head"].values, marker="o", c='g', s=20,alpha=0.5, label="interpolated")

    plt.tight_layout()
    plt.savefig(os.path.join(ws,"cleaned_head.pdf"))
    plt.close(fig)

    foutput_df.dropna(inplace=True)
    foutput_df.pop("idx")
    foutput_df.to_csv(os.path.join(ws,"heads.csv"))

    ml = test_pastas_model(ws)
    setup_pst(ws)
    # input_df = get_input_df(ws)
    # input_df.index.name = "datetime"
    # input_df.to_csv(os.path.join(ws, "input_data.csv"))
    # output_df = get_output_df(ws)
    # ml = get_pastas_model(input_df, output_df)
    # ml.simulate()
    # ml.del_transform()
    # ml.solve()
    #ml.plot()
    #plt.show()

def setup_smoothed_obs(t_d):
    """prep the smoothed simulated outputs
    """
    b_d = os.getcwd()
    os.chdir(t_d)
    df = process_smoothed_obs()
    os.chdir(b_d)
    return df
    # output_df = get_output_df(t_d)
    # df.plot(subplots=True,sharey=True)
    # output_df.plot()
    # plt.show()



def process_smoothed_obs():
    """process the sim results into smoothed results
    """
    results = pd.read_csv("results.csv",index_col=0,parse_dates=[0])
    df = smooth(results)
    df.to_csv("smoothed.csv")

    return df


def smooth(df):
    """smoothing function
    """
    windows = [365 * i for i in range(1,5)]
    
    windows.insert(0,28)
    windows.insert(0,7)

    rolled = []
    for window in windows:
        roll = df.gwsim.rolling(window).mean().values
        rolled.append(roll)
    names = ["roll-{0:04.0f}".format(window) for window in windows]
    results_df = pd.DataFrame(rolled, columns=df.index, index=names).T
    results_df.index.name = "datetime"
    results_df.dropna(inplace=True)
    return results_df


    #import PyEMD
    #eemd = PyEMD.EEMD()
    #r = eemd(results.gwsim.values)


def gather_pdfs(dest_d):
    if os.path.exists(dest_d):
        shutil.rmtree(dest_d)
    os.makedirs(dest_d)
    dirs = [os.path.join("models",d) for d in os.listdir("models") if os.path.isdir(os.path.join("models",d))]
    for d in dirs:
        pdfs = [f for f in os.listdir(d) if f.endswith(".pdf")]
        for pdf in pdfs:
            shutil.copy2(os.path.join(d,pdf),os.path.join(dest_d,pdf))
            print(pdf)

def plot_results_all_masters_mp():
    m_ds = [os.path.join("models", d) for d in os.listdir("models") if
            os.path.isdir(os.path.join("models", d)) and "master" in d]
    procs = []
    for m_d in m_ds:
        p = Process(target=plot_results,args=(m_d,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def setup_verf_template(org_t_d,start_dt,end_dt,new_t_d,daily_par_factor=1.0):
    """modify the weights in the control file to have an unseen verification period
    """
    assert os.path.exists(org_t_d)
    if os.path.exists(new_t_d):
        shutil.rmtree(new_t_d)
    shutil.copytree(org_t_d,new_t_d)
    pst = pyemu.Pst(os.path.join(new_t_d,"pest.pst"))
    obs = pst.observation_data.loc[pst.nnz_obs_names,:].copy()
    print(obs.obgnme.value_counts())
    org_nnz = pst.nnz_obs
    dobs = obs.loc[pd.notna(obs.datetime),:].copy()
    dobs.loc[:,"datetime"] = pd.to_datetime(dobs.datetime)
    keep_dobs = dobs.loc[dobs.datetime.apply(lambda x: x < start_dt or x >= end_dt),:].copy()
    assert keep_dobs.shape[0] > 0
    holdback_dobs = dobs.loc[dobs.datetime.apply(lambda x: x >= start_dt and x < end_dt),:].copy()
    obs = pst.observation_data
    obs.loc[:,"verf_holdback"] = False
    obs.loc[:,"org_weight"] = obs.weight.values.copy()    
    obs.loc[holdback_dobs.obsnme,"verf_holdback"] = True

    print(dobs.shape[0],keep_dobs.shape[0],holdback_dobs.shape[0])
    assert obs.loc[obs.verf_holdback==True,:].shape[0] == dobs.shape[0] - keep_dobs.shape[0]

    obs.weight.loc[dobs.obsnme.values] = 0.0
    obs.weight.loc[keep_dobs.obsnme] = keep_dobs.weight.values

    print(obs.loc[pst.nnz_obs_names,"obgnme"].value_counts())
    print("non-zero weighted obs reduced from ",org_nnz," to ",pst.nnz_obs)
    #revise noise en
    noise = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=os.path.join(new_t_d,"noise.jcb"))
    noise = noise.loc[:,pst.nnz_obs_names]
    noise.to_binary(os.path.join(new_t_d,"noise.jcb"))

    #fix daily pars in that verf period
    par = pst.parameter_data
    daily_pnames = ["tmean", "pet", "precip"]
    if "stage" in par.pname.unique():
        daily_pnames.append('stage')
    daily_pnames = set(daily_pnames)
    dpar = par.loc[par.pname.isin(daily_pnames), :].copy()
    dpar.loc[:, "datetime"] = pd.to_datetime(dpar.datetime)
    holdback_dpar = dpar.loc[dpar.datetime.apply(lambda x: x >= start_dt and x < end_dt),:].copy()
    org_nadj = pst.npar_adj
    par.loc[holdback_dpar.parnme,"partrans"] = "fixed"
    assert pst.npar_adj < org_nadj
    print("adj pars reduced from ",org_nadj," to ",pst.npar_adj)

    if daily_par_factor != 1.0:
        fixed_dpar = par.loc[par.apply(lambda x: x.partrans=="fixed" and x.pname in daily_pnames,axis=1),:]
        assert fixed_dpar.shape[0] > 0

        pe = pyemu.ParameterEnsemble.from_binary(pst=pst,filename=os.path.join(new_t_d,"prior.jcb"))
        dpe = pe._df.loc[:,fixed_dpar.parnme].apply(np.log10)
        
        print("before mean",dpe.mean(axis=1).values)
        print("before std",dpe.std(axis=1).values)
        dpe.loc[:,:] *= daily_par_factor
        print("after mean",dpe.mean(axis=1).values)
        print("after std",dpe.std(axis=1).values)
        pe.loc[:,dpe.columns] = 10**dpe.values
        pe.to_binary(os.path.join(new_t_d,"prior.jcb"))


    pst.control_data.noptmax = -2
    pst.write(os.path.join(new_t_d,"pest.pst"),version=2)
    pyemu.os_utils.run("pestpp-ies pest.pst",cwd=new_t_d)


def get_verf_end_dt(start_dt,num_days):
    """get the verfication end date
    """
    end_dt = start_dt + pd.to_timedelta(num_days,unit="d")
    return end_dt

def plot_conf_lim_pareto(m_d,keep_best=10000):
    pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
    pst.try_parse_name_metadata()
    obs = pst.observation_data

    gwobs = obs.loc[obs.apply(lambda x: x.usecol=="gwsim" and x.observed==True,axis=1),:].copy()
    assert gwobs.shape[0] > 0
    hobs = gwobs.loc[obs.verf_holdback==False,:].copy()
    assert hobs.shape[0] > 0
    vobs = gwobs.loc[obs.verf_holdback==True,:].copy()
    assert vobs.shape[0] > 0
    
    #pe_dict = {}
    oe_dict = {}
    iters = []

    for i in range(pst.control_data.noptmax+1):
        pe_file = os.path.join(m_d,"pest.{0}.par.jcb".format(i))
        oe_file = os.path.join(m_d, "pest.{0}.obs.jcb".format(i))

        if os.path.exists(pe_file) and os.path.join(oe_file):
            iters.append(i)

    if 0 not in iters:
        iters.insert(0,0)

    for i in iters:
        pe_file = os.path.join(m_d, "pest.{0}.par.jcb".format(i))
        oe_file = os.path.join(m_d, "pest.{0}.obs.jcb".format(i))
        oe = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=oe_file)
        #pe = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=pe_file)
        if keep_best < 0:
            kkeep_best = int(oe.shape[0] * np.abs(keep_best))
        else:
            kkeep_best = keep_best
        if oe.shape[0] > kkeep_best:
            pv = oe.phi_vector
            pv.sort_values(inplace=True,ascending=True)
            keep = pv.iloc[:kkeep_best].index.values
            oe = oe.loc[keep,:]
            #pe = pe.loc[keep,:]

        #pe_dict[i] = pe
        oe_dict[i] = oe

    #pr_pe = pe_dict.pop(0)
    #pr_oe = oe_dict.pop(0)

    lb_pt_vals = np.arange(1,50,1)
    ub_pt_vals = 100 - lb_pt_vals

    ppairs = [[lb,ub] for lb,ub in zip(lb_pt_vals,ub_pt_vals)]
    labels = ["{0} - {1}".format(p[0],p[1]) for p in ppairs]
    gvals = np.arange(len(ppairs))

    with PdfPages(os.path.join(m_d,"verf_pareto.pdf")) as pdf:
        hpv_vecs,vpv_vecs = [],[]
        hbase,vbase = [],[]
        for it in iters:
            oe = oe_dict[it]#.loc[:,gwobs.obsnme]
            obs = oe.pst.observation_data
            obs.weight = 0.0
            obs.weight.loc[hobs.obsnme] = hobs.org_weight.values
            hpv = oe.phi_vector
            obs = oe.pst.observation_data
            obs.weight = 0.0
            obs.weight.loc[vobs.obsnme] = vobs.org_weight.values
            vpv = oe.phi_vector
            hpv_vecs.append(np.log10(hpv.values))
            vpv_vecs.append(np.log10(vpv.values))
            hbase.append(np.log10(hpv.loc["base"]))
            vbase.append(np.log10(vpv.loc["base"]))
           
            fig,axes = plt.subplots(2,2,figsize=(10,10))
            axes[1,0].scatter(hpv.values,vpv.values,marker=".",c="0.5",alpha=0.5)
            axes[1,0].scatter(hpv.values.mean(),vpv.values.mean(),marker="o",c="m",s=20)

            
            axes[0,1].scatter(vpv.values,hpv.values,marker=".",c="0.5",alpha=0.5)
            axes[0,1].scatter(vpv.values.mean(),hpv.values.mean(),marker="o",c="m",s=20)
            if "base" in hpv.index and "base" in vpv.index:
                axes[1,0].scatter(hpv.loc["base"],vpv.loc["base"],marker="o",c="b",s=20)
                axes[0,1].scatter(vpv.loc["base"],hpv.loc["base"],marker="o",c="b",s=20)
            
            axes[0,0].hist(hpv.values,facecolor="0.5",bins=20)
            axes[0,0].set_xlim(axes[1,0].get_xlim())
            axes[1,1].hist(vpv.values,facecolor="0.5",bins=20)
            axes[1,1].set_xlim(axes[0,1].get_xlim())
            axes[0,0].set_title("historic phi iteration {0}".format(it))
            axes[1,1].set_title("verf phi")
            axes[0,1].set_xlabel("verf")
            axes[1,0].set_xlabel("historic")
            axes[1,0].set_ylabel("verf")
            axes[0,1].set_ylabel("historic")
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
            

        fig,ax = plt.subplots(1,1,figsize=(6,6))
        vparts = ax.violinplot(hpv_vecs, iters,showextrema=False)
        for pc in vparts['bodies']:
            pc.set_facecolor('c')
            pc.set_edgecolor('none')
            pc.set_alpha(0.5)
        axt = plt.twinx(ax)
        vparts = axt.violinplot(vpv_vecs, iters,showextrema=False)
        for pc in vparts['bodies']:
            pc.set_facecolor('m')
            pc.set_edgecolor('none')
            pc.set_alpha(0.5)
        ax.scatter(iters,hbase,marker=".",c="c",s=20)
        ax.scatter(iters,vbase,marker=".",c="m",s=20)
        
        ax.set_xlabel("iteration")
        ax.set_ylabel("historic phi",color="c")
        axt.set_ylabel("verf phi",color="m")
        ax.set_title("historic phi (cyan) and verification phi (magenta)",loc="left")
        ymn = min(ax.get_ylim()[0],axt.get_ylim()[0])
        ymx = min(ax.get_ylim()[1],axt.get_ylim()[1])
        ax.set_ylim(ymn,ymx)
        axt.set_ylim(ymn,ymx)
        ax.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)
        

        for it in iters:
            oe = oe_dict[it].loc[:,vobs.obsnme]
            vals = oe.values
            total_misses = []
            for ppair in ppairs:
                misses = 0
                for ival,vval in enumerate(vobs.obsval):
                    llb,uub = np.percentile(vals[:,ival],ppair)
                    if vval < llb or vval > uub:
                        misses += 1
                total_misses.append(misses)
            fig,ax = plt.subplots(1,1,figsize=(10,10))
            ax.plot(gvals,total_misses)
            ax.set_xticks(gvals)
            ax.set_xticklabels(labels,rotation=90)
            y = vobs.shape[0] * 0.05
            ax.plot(ax.get_xlim(),[y,y],"r--",lw=1.5)
            ax.set_ylabel("misses")
            ax.set_xlabel("interval")
            ax.set_title("iteration: {0}, num of verf values: {1}".format(it,vobs.shape[0]),loc="left")
            ax.grid()
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)

        

def build_submission():
    """build the final submission files
    """
    m_ds = [os.path.join(d) for d in os.listdir("models") if "master" in d and "verf1" not in d and "org" not in d and os.path.isdir(os.path.join("models",d))]
    sites = [d.split('_')[0] for d in m_ds]
    # which iteration to use for each site - subjective
    iter_dict = {"usa":5,"netherlands":8,"sweden1":3,"sweden2":6,"germany":5}
    # which percentiles to use for the confidence limit - subjective
    cl_dict = {"usa":[10,90],"netherlands":[10,90],"sweden1":[12,92],"sweden2":[20,75],"germany":[20,75]}
    # number of realizations to use in the confidence limit calculations
    keep_dict = {s:150 for s in sites}

    sub_dir = "org_submission_forms"
    sub_files = os.listdir(sub_dir)
    sub_sites = ["".join(s.lower().split('.')[0].split("_")[2:]) for s in sub_files]
    sub_dict = {s:f for s,f in zip(sub_sites,sub_files)}
    fsub_dir = "final_submission_forms"
    if os.path.exists(fsub_dir):
        shutil.rmtree(fsub_dir)
    os.makedirs(fsub_dir)

    for site,m_d in zip(sites,m_ds):

        subdf = pd.read_csv(os.path.join(sub_dir,sub_dict[site]))
        subdf.loc[:,"Date"] = pd.to_datetime(subdf.Date)
        subdf.index = subdf.Date

        
        m_d = os.path.join("models",m_d)
        pst = pyemu.Pst(os.path.join(m_d,"pest.pst"))
        pst.try_parse_name_metadata()
        obs = pst.observation_data

        gwobs = obs.loc[obs.usecol=="gwsim",:].copy()
        gwobs.loc[:,"datetime"] = pd.to_datetime(gwobs.datetime)
        gwobs.index = gwobs.datetime
        gwobs = gwobs.loc[subdf.index,:]
        gwobs.index = gwobs.obsnme
              
        oe_file = os.path.join(m_d, "pest.{0}.obs.jcb".format(iter_dict[site]))
            
        oe = pyemu.ObservationEnsemble.from_binary(pst=pst,filename=oe_file)
        keep_best = keep_dict[site]
        if keep_best < 0:
            kkeep_best = int(oe.shape[0] * np.abs(keep_best))
        else:
            kkeep_best = keep_best
        if oe.shape[0] > kkeep_best:
            pv = oe.phi_vector
            pv.sort_values(inplace=True,ascending=True)
            keep = pv.iloc[:kkeep_best].index.values
            oe = oe.loc[keep,:]
        vals = oe.loc[:,gwobs.obsnme].values
        pvals = cl_dict[site]
        for ival,name in enumerate(gwobs.obsnme):
                    llb,uub = np.percentile(vals[:,ival],pvals)
                    gwobs.loc[name,"lb"] = llb
                    gwobs.loc[name,"ub"] = uub
        
        # use the naive estimate that the netherlands site has a strong upper limit control on 
        # gw levels to limit the upper end of the confidence limit
        if "netherlands" in site:
            gwobs.loc[gwobs.ub>11.4,"ub"] = 11.4 
                    
        subdf.loc[:,"Simulated Head"] = oe.loc["base",gwobs.obsnme].values
        subdf.loc[:,"95% Lower Bound"] = gwobs.loc[:,"lb"].values
        subdf.loc[:,"95% Upper Bound"] = gwobs.loc[:,"ub"].values
        subdf.pop("Date")
        subdf.to_csv(os.path.join(fsub_dir,sub_dict[site]))
        fig,ax = plt.subplots(1,1,figsize=(15,5))
        ax.plot(subdf.index,subdf.loc[:,"Simulated Head"].values,"b-",label="base",lw=1.5)
        ax.fill_between(subdf.index.values,subdf.loc[:,"95% Lower Bound"].values,subdf.loc[:,"95% Upper Bound"].values,facecolor="b",alpha=0.5)
        gwobs = gwobs.loc[gwobs.observed==True,:]
        ax.scatter(gwobs.datetime.values,gwobs.obsval.values,marker="^",c="r",s=10,label="observed")
        ax.set_title(site,loc="left")
        ax.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(fsub_dir,site+".pdf"))
        plt.close(fig)
        plot_results(m_d, keep_best=1000,post_iter=iter_dict[site],percentile_vals=cl_dict[site],plt_name=os.path.join(fsub_dir,"sub_results_{0}.pdf".format(site)))
        plot_results(m_d+"_verf1", keep_best=1000,post_iter=iter_dict[site],percentile_vals=cl_dict[site],plt_name=os.path.join(fsub_dir,"sub_results_verf1_{0}.pdf".format(site)))

        





if __name__ == "__main__":

    num_reals = 300
    noptmax = 10
    num_workers=20

    prep_funcs= [prep_ned,prep_sweden1,prep_germany,prep_usa,prep_sweden2]
    names = ["netherlands","sweden1","germany","usa","sweden2"]
    
    fac_dict = {'usa':0.5,"netherlands":0.85,"germany":0.5,"sweden1":0.8,"sweden2":1.0}

    for prep_func, name in zip(prep_funcs,names):
        prep_func()
        t_d = os.path.join("models", "{0}_template".format(name))
        m_d = os.path.join("models", "{0}_master".format(name))
        run(t_d,m_d=m_d,
        num_workers=num_workers, noptmax=noptmax, ies_num_reals=num_reals,
        ies_bad_phi_sigma=1.5, ies_init_lam=-10)
        plot_results(m_d, post_iter=None)

        start_dt = pd.to_datetime("1-1-2008")
        end_dt = get_verf_end_dt(start_dt, 365 * 4)
        new_t_d = t_d + "_verf1"
        setup_verf_template(t_d, start_dt, end_dt, new_t_d, daily_par_factor=fac_dict[name])
        m_d = m_d + "_verf1"
        run(new_t_d, m_d=m_d,
          num_workers=num_workers, noptmax=noptmax, ies_num_reals=num_reals,
          ies_bad_phi_sigma=1.5, ies_init_lam=-10,panther_agent_freeze_on_fail=False)
        plot_conf_lim_pareto(m_d)
        plot_results(m_d, keep_best=1000,post_iter=None,percentile_vals=[5,95],plt_name=None)
        
    build_submission()

    #plot_results_all_masters_mp()

    #gather_pdfs("results")
