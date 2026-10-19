********************************************************************************
* STANDALONE: Rambachan & Roth (2023) HonestDiD Test
* Copy-paste this entire file into Stata and run
********************************************************************************

clear all
set more off

* ---- INSTALL PACKAGES (run once) ----
ssc install honestdid, replace
ssc install reghdfe, replace  
ssc install ftools, replace

* ---- LOAD DATA ----
import delimited "master_state_year_2007_2013.csv", clear

* ---- PREP ----
destring *, replace force
xtset state_code year

* ---- CREATE EVENT-TIME INTERACTIONS ----
* We need: evt_oxy_m3 evt_oxy_m2 [omit m1=baseline] evt_oxy_p0 evt_oxy_p1 evt_oxy_p2 evt_oxy_p3

gen evt_m3 = (event_time == -3)
gen evt_m2 = (event_time == -2)
gen evt_p0 = (event_time ==  0)
gen evt_p1 = (event_time ==  1)
gen evt_p2 = (event_time ==  2)
gen evt_p3 = (event_time ==  3)

gen evt_oxy_m3 = evt_m3 * initial_oxy
gen evt_oxy_m2 = evt_m2 * initial_oxy
gen evt_oxy_p0 = evt_p0 * initial_oxy
gen evt_oxy_p1 = evt_p1 * initial_oxy
gen evt_oxy_p2 = evt_p2 * initial_oxy
gen evt_oxy_p3 = evt_p3 * initial_oxy

* ---- RUN EVENT STUDY ----
* IMPORTANT: the order of coefficients matters for honestdid
* They must be listed pre-treatment first, then post-treatment, 
* in chronological order

reghdfe fertility_rate_proxy_per_1000_women_15_44 ///
    evt_oxy_m3 evt_oxy_m2 ///
    evt_oxy_p0 evt_oxy_p1 evt_oxy_p2 evt_oxy_p3 ///
    xxunemp ln_hpi ln_pop ///
    share_young share_old share_white share_low_edu ///
    pp_pmp_authorized pp_must, ///
    absorb(state_code year) cluster(state_code)

* Check coefficients look right
estimates store es_main

* ---- HONESTDID: SMOOTHNESS RESTRICTION (delta=sd) ----
* This is the main R&R test
* numpre(2) = number of pre-treatment coefficients (m3, m2)
* numpost(4) = number of post-treatment coefficients (p0, p1, p2, p3)
* mvec = values of M to test (0 = exact parallel trends)

honestdid, numpre(2) numpost(4) mvec(0(0.25)2.0) delta(sd)

* This outputs:
*   - For each M: the robust confidence interval for the treatment effect
*   - The breakdown M where the CI first includes zero

* ---- HONESTDID: RELATIVE MAGNITUDES (delta=rm) ----
* Alternative approach: bounds the ratio of post-treatment bias to 
* the max pre-treatment violation, without specifying M directly

honestdid, numpre(2) numpost(4) delta(rm) mvec(0(0.25)2.0)

* ---- GENERATE THE SENSITIVITY PLOT ----
* This creates the classic R&R figure showing CI width vs M

honestdid, numpre(2) numpost(4) mvec(0(0.25)2.0) delta(sd) coefplot
graph export "honestdid_sensitivity.png", replace width(1200)

* ---- INTERPRETATION ----
* Look at the output table. Find the row where the CI first includes 0.
* That M value is your "breakdown value."
*
* Example interpretation for the paper:
*
* "We implement the sensitivity analysis of Rambachan and Roth (2023) 
*  to assess how robust our findings are to violations of the parallel 
*  trends assumption. Under exact parallel trends (M=0), the 95% 
*  robust confidence interval for the average post-treatment effect 
*  is [X, Y]. The breakdown value of M is Z, meaning that the slope 
*  of the differential trend would need to change by at least Z units 
*  per period — substantially more than the [max pre-period change] 
*  observed in the data — to overturn our finding of a positive effect."

* ---- ALSO RUN THE WEIGHTED VERSION ----
reghdfe fertility_rate_proxy_per_1000_women_15_44 ///
    evt_oxy_m3 evt_oxy_m2 ///
    evt_oxy_p0 evt_oxy_p1 evt_oxy_p2 evt_oxy_p3 ///
    xxunemp ln_hpi ln_pop ///
    share_young share_old share_white share_low_edu ///
    pp_pmp_authorized pp_must ///
    [aweight=pop], ///
    absorb(state_code year) cluster(state_code)

estimates store es_weighted

honestdid, numpre(2) numpost(4) mvec(0(0.25)2.0) delta(sd) coefplot
graph export "honestdid_sensitivity_weighted.png", replace width(1200)
