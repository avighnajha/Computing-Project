#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from operator import itemgetter
import time
start_time = time.time()

spectral_data = np.loadtxt("Halpha_spectral_data.csv", skiprows = 4, delimiter = ",")
spectral_data = np.array(sorted(spectral_data, key=itemgetter(0))) 
#This line of code used to sort the 2 dimensional array by the first column is from https://stackoverflow.com/questions/20099669/sort-multidimensional-array-based-on-2nd-element-of-the-subarray

row1 =spectral_data[0,:]
freq_all = row1[1:]

distance_data = np.loadtxt("Distance_Mpc.txt", skiprows = 1)
distance_data= np.array(sorted(distance_data, key=itemgetter(0)))


trash = [] #This list will contain the observation numbers of all of the observations with a bad valid instrument response 

for i in distance_data:
    if i[2] == 0:
        trash.append(i[0])



#filtering spectral_data to get rid of inaccurate values 
for j in trash:
    for i in range(len(spectral_data)):
        if spectral_data[i,0] == j:
            spectral_data = np.delete(spectral_data, i,0)
            break
spectral_data = spectral_data[1:]

clean_distance = [] #This list will contain all of the distance data with a valid isntrument response of 1
for i in distance_data:
    if i[2] == 1:
        clean_distance.append(i)


#%%
#Plotting intensity against frequency 

params = {
        "figure.figsize": [20,40]
        }

plt.rcParams.update(params) 

#Function to create a guess for the gradient and y intercept of the straight line fit by finding the gradient and y-interecept of a line through the first and the last point.
def guess_m_c( x_1, x_2,y_1, y_2):
    fit = np.polyfit([x_1, x_2],[y_1, y_2], 1, cov = False )
    m = fit[0] #m is the guess for the gradient
    c = fit[1] #c is the guess for the y-intercept
    return m, c, fit

#Function to generate guesses for the parameters a and mu by finding the x coordinate of the point farthest away from the straight line and the distance between the approximate straght line and the gaussian feature
def guess_a_mu(fit, x, y):
    fitfunc = np.poly1d(fit)
    y_line = fitfunc(x)
    dif = y-y_line
    a = max(dif)
    max_idx = np.argmax(dif)
    #The above 2 lines of code to get the max value for the difference between y and y_line is from https://stackoverflow.com/questions/3989016/how-to-find-all-positions-of-the-maximum-value-in-a-list
    mu = x[max_idx]
    return a, mu

#The function to fit the gaussian feature 
def fit_func(x, a, mu, sig, m, c):
    line = m*x + c
    gauss = a*np.exp((-(x-mu)**2)/(2*sig**2))
    return line+gauss

x_max_all = []
x_max_uncall = []

n =1
#This loop goes through all the 25 valid sets of data and plots their intensities against the frequencies, generates the gaussian fit, finds guesses for the parameters for plotting the gaussian fit and finds the frequncies at which the intensity peaks.
for i in spectral_data:
    plt.subplot(15,2,n)
    plt.plot(freq_all, i[1:], "x")
    n+=1
    m,c, fit = guess_m_c(freq_all[0], freq_all[-1],i[1], i[-1])
    a, mu = guess_a_mu(fit, freq_all, i[1:])
    sig = 0.15*10**14 #Guessed through inspection of the graphs
    guesses = [a, mu, sig, m,c ]
    params, covariant = curve_fit(fit_func, freq_all, i[1:],guesses, maxfev = 5000)
    print(params)
    data_fit = fit_func(freq_all, *params)
    #plt.plot(freq_all, data_fit)
    #plt.show()
    
    x_max = params[1]
    x_max_all.append(x_max)
    x_max_unc = np.sqrt(covariant[1][1])
    x_max_uncall.append(x_max_unc)

#%%
#Converting frequency data to wavelength 
c = 2.9979*10**8
lambda_o = []
lambda_o_unc = []
#This loop converts the pobserved frrequency values into observed wavelength and also gets the uncertainties in the observed wavelength

for i in range(len(x_max_all)):
    lambda_o.append(c/x_max_all[i])
    lambda_o_unc.append((x_max_uncall[i]*c)/(x_max_all[i]**2))



#%%
#Finding the velocity of the galaxies using the redshift formula
velocity = []
lambda_e = 656.28*10**-9
v_unc = []
for i in range(len(lambda_o)):
    v = c*(lambda_o[i]**2-lambda_e**2)/(lambda_o[i]**2+lambda_e**2)
    velocity.append(v)
    v_unc.append(((4*lambda_o[i]*c*lambda_e**2)/(lambda_o[i]**2+lambda_e**2)**2)*lambda_o_unc[i])

v_unc = np.array(v_unc)

#%%
velocity = np.array(velocity)
clean_distance = np.array(clean_distance)
plt.style.use("default")

#Scatter plot for the velocity against distance
#plt.scatter(clean_distance[:,1], velocity, color = "red")

vd_fit, cov_fit = np.polyfit(clean_distance[:,1], velocity, 1,w = 1/v_unc, cov =True)
vd_fit = vd_fit.flatten()

vd_fit_func = np.poly1d(vd_fit)
#Plotting the line of best fit
#plt.plot(clean_distance[:,1], vd_fit_func(clean_distance[:,1]))
#plt.xlabel("Distance (Mpc)")
#plt.ylabel("Velocitty of galaxies (m/s)")
#plt.legend()
#plt.savefig("Velocity vs Distance.png")
#plt.show()

H_0 = vd_fit[0]/1000
unc_H_0 = np.sqrt(cov_fit[0][0])/1000

print("Therefore the value for Hubble's constant H_0 is %.1f +/- %.1f km/s/Mpc"%(H_0, unc_H_0))


print("My program took", time.time() - start_time, "to run")


















