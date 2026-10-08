import numpy as np
from scipy import fft, special, ndimage


def hermite(X, N):              #hermite polynomial
    HER = special.hermite(N)
    sn = HER(X)
    return sn

def laguerre(X, L, P):          #laguerre polynomial
    LAG = special.genlaguerre(P, L)
    sn = LAG(X)
    return sn


class Space():
    def __init__(self, nix, Dx, niy=None, Dy=None, sparse=True):
        """ Initiate an object with the necessary parameters for calculating transverse fields
        """
        self.sparse = sparse
        self.nix = nix
        if niy == None:
            self.niy = nix
        else:
            self.niy = niy
        self.Dx = Dx
        if Dy == None:
            self.Dy = Dx
        else:
            self.Dy = Dy
        self.x, self.y = np.meshgrid(np.linspace(-self.nix, self.nix, self.Dx), np.linspace(-self.niy, self.niy, self.Dy), sparse=sparse)
        self.field = np.zeros((self.Dy, self.Dx), dtype='complex128')
        self.kx, self.ky = np.meshgrid(2*np.pi*fft.fftfreq(self.Dx, 2*self.nix/self.Dx), 2*np.pi*fft.fftfreq(self.Dy, 2*self.niy/self.Dy), sparse=sparse)
        self.lamb = 1064e-9        #standard 1064 nanometer         
        self.waist = 1e-6          #standard 1 micrometer
        self.x0 = 0
        self.y0 = 0

    def __mul__(self, other):
        """Multiply the transverse fields point by point if multiplied by another field, or globally if multiplied by number.
        Changes lamb by 1/lamb_new = 1/lamb_1 + 1/lamb_2. Doesn't change waist.
        
        """
        if isinstance(other, (type(self))):
            new = Space(self.nix, self.Dx, niy=self.niy, Dy=self.Dy, sparse=self.sparse)
            new.field = self.field*other.field
            new.lamb = 1/(1/self.lamb + 1/other.lamb)
            new.waist = self.waist
            new.x0 = self.x0
            new.y0 = self.y0
            return new
        if isinstance(other, (int, float, complex)):
            new = Space(self.nix, self.Dx, niy=self.niy, Dy=self.Dy, sparse=self.sparse)
            new.field = other*self.field
            new.lamb = self.lamb
            new.waist = self.waist
            new.x0 = self.x0
            new.y0 = self.y0
            return new

        else:
            raise TypeError(f"sorry, don't know how to multiply by {type(other).__name__}")
        
    __rmul__ = __mul__
    
    def __add__(self, other):
        if isinstance(other, (Space, type(self))):
            new = Space(self.nix, self.Dx, niy=self.niy, Dy=self.Dy, sparse=self.sparse)
            new.field = other.field + self.field
            new.lamb = self.lamb
            new.waist = self.waist
            new.x0 = self.x0
            new.y0 = self.y0
            return new
        else:
            raise TypeError(f"sorry, don't know how to add by {type(other).__name__}")
        
    __radd__ = __add__
        
    def __sub__(self, other):
        if isinstance(other, Space):
            new = Space(self.nix, self.Dx, niy=self.niy, Dy=self.Dy, sparse=self.sparse)
            new.field = self.field - other.field
            new.lamb = self.lamb
            new.waist = self.waist
            new.x0 = self.x0
            new.y0 = self.y0
            return new
        else:
            raise TypeError(f"sorry, don't know how to subtract by {type(other).__name__}")
        
    __rsub__ = __sub__

    def zr(self):
        return np.pi*self.waist**2/self.lamb
        
    def hg(self, N, M, z=0):                                                        #get a HG mode at distance z of order N+M
        zr = self.zr()
        q0 = 1j*zr
        q = z + 1j*zr
        k = 2*np.pi/self.lamb
        w = self.waist*np.sqrt(1 + (z/zr)**2)
        Cn = np.sqrt(np.sqrt(2/np.pi)*q0/(2**N * np.math.factorial(N) * q * self.waist))
        un = Cn*(-np.conjugate(q)/q)**(N/2)*hermite(np.sqrt(2)*(self.x-self.x0)/w, N)*np.exp(-1j*k*(self.x-self.x0)**2/(2*q))
        Cm = np.sqrt(np.sqrt(2/np.pi)*q0/(2**M * np.math.factorial(M) * q * self.waist))
        um = Cm*(-np.conjugate(q)/q)**(M/2)*hermite(np.sqrt(2)*(self.y-self.y0)/w, M)*np.exp(-1j*k*(self.y-self.y0)**2/(2*q))
        self.field = un*um*np.exp(-1j*k*z)
        return self
    
    def lg(self,l,p, z=0):                                    #get a LG mode at distance z of order abs(N) + 2M
        zr = self.zr()
        k = 2*np.pi/self.lamb
        w = self.waist*np.sqrt(1 + (z/zr)**2)
        R = (z**2 + zr**2)
        gouy = (np.abs(l) + 2*p + 1)*np.arctan(z/zr)
        r = np.sqrt((self.x-self.x0)**2 + (self.y-self.y0)**2)
        Cte = np.sqrt(2*np.math.factorial(p) / (np.pi*np.math.factorial(p+np.abs(l))))
        self.field = Cte*(1/w)*(np.sqrt(2)*r/w)**(np.abs(l)) * np.exp(-(r/w)**2) * laguerre(2*(r/w)**2, np.abs(l), p) *\
            np.exp(-1j*(k*z + k*z*(r**2)/(2*R) + l*np.arctan2(self.y-self.y0,self.x-self.x0) - gouy))
        return self
    
    def bessel(self, N, theta, z=0):                                    #get a bessel beam of order N scaled to meter order of magnitude, with angle of conic wave M.
        k = 2*np.pi/self.lamb
        kz = k*np.cos(theta)
        kr = k*np.sin(theta)
        self.field = np.exp(1j*kz*z)*special.jv(np.abs(N),np.sqrt((self.x-self.x0)**2 + (self.y-self.y0)**2)/self.waist**2)*\
            np.exp(1j*N*np.arctan2(self.y-self.y0, self.x-self.x0))
        return self
    
    def gbessel(self, N, r0):                                    #get a gaussian bessel beam of order N at z=0, M is the asymptotic radial period
        rad = 2*np.pi*self.waist**2/r0
        r = np.sqrt((self.x-self.x0)**2 + (self.y-self.y0)**2)
        self.field = special.jv(np.abs(N),rad*r/self.waist**2)*\
            np.exp(1j*N*np.arctan2(self.y-self.y0, self.x-self.x0))*np.exp(-(r/self.waist)**2)
        return self

    def sup(self, base, coefs, z=0):                          #get a supperposition of HG modes or LG modes from a list of coefficients. It runs the list ordered by  
        norm = np.sqrt(np.sum(np.abs(coefs)**2))             #dimension, decreasing n and increasing m for HG modes, and first increasing l then increasing p for LG modes.
        coefs = coefs/norm
        num = len(coefs)
        c = c2 = dim = c3 = 0
        while c < num :
            c += 1
            c2 += c
            if c2 < num:
                dim +=1
        if base == "H":
            for d in range(dim+1):
                for j in range(dim+1):
                    for i in range(dim+1):
                        if i + j == d:
                            if c3<num:
                                self.field += coefs[c3]*(self.hg(i, j, z = z).field)
                                c3 += 1
        if base == "L":
            for d in range(dim+1):
                for j in range(dim+1):
                    for i in range(-(dim), dim+1):
                        if np.abs(i) + 2*j == d:
                            if c3<num:
                                self.field += coefs[c3]*(self.lg(i, j, z = z).field)
                                c3 += 1
        self.coefs = coefs
        return self
    
    def frac_oam(self, Ma, n_modes, beta = 0, theta_0=0, z = 0):        #get a fractional OAM beam, with OAM Ma (!= integer), by the method of LG supperpositions.
        mu = Ma%1
        m = Ma-mu
        if mu >=0.5:
            n_min = np.ceil(Ma - n_modes / 2)
        else: 
            n_min = np.floor(Ma - n_modes / 2)
        n_max = n_min + n_modes - 1
        coefs = []
        for l in np.arange(int(n_min), int(n_max)+1):
            coef = np.exp(-1j*mu*beta)*1j*np.exp(1j*(Ma-l)*theta_0)/(2*np.pi*(Ma-l))*np.exp(1j*(m-l)*beta)*(1-np.exp(1j*mu*2*np.pi))
            self.field += coef*(self.get_lg(l, 0, z = z).field)
            coefs.append(np.abs(coef)**2)
        self.coefs= np.column_stack((np.arange(n_min, n_max+1),coefs)) 
        return self
                        
    def laguerre_radial(self, l, lr, p, z = 0):
        zr = self.zr()
        k = 2*np.pi/self.lamb
        w = self.w0*np.sqrt(1 + (z/zr)**2)
        R = (z**2 + zr**2)
        gouy = (np.abs(l) + 2*p + 1)*np.arctan(z/zr)
        r = np.sqrt((self.x-self.x0)**2 + (self.y-self.y0)**2)
        self.field = (self.w0/w)*(np.sqrt(2)*r/w)**(np.abs(lr)) * np.exp(-(r/w)**2) * laguerre(2*(r/w)**2, l, p) * \
            np.exp(-1j*(k*z + k*z*((self.x-self.x0)**2 + (self.y-self.y0)**2)/(2*R) - l*np.arctan2((self.y-self.y0),(self.x-self.x0)) + 1j*gouy))
        self.norm_beam()
        return self
    
    def get_Power(self):
        """Get the total Power of a field within Space"""
        return np.sum(np.abs(self.field)**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def hhalf1(self):
        """Def horizontal half 1, subsequent functions do the same for other halfs """
        hhalf1 = (self.x >= 0)
        return self.field*hhalf1
    
    def hhalf2(self):
        hhalf2 = (self.x < 0)
        return self.field*hhalf2
    
    def vhalf1(self):
        vhalf1 = (self.y >= 0)
        return self.field*vhalf1
    
    def vhalf2(self):
        vhalf2 = (self.y < 0)
        return self.field*vhalf2
    
    def get_Power_hhalf1(self):
        """Get Power of horizontal half 1""" 
        return np.sum(np.abs(self.hhalf1())**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def get_Power_hhalf2(self):
        return np.sum(np.abs(self.hhalf2())**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def get_Power_vhalf1(self):
        return np.sum(np.abs(self.vhalf1())**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def get_Power_vhalf2(self):
        return np.sum(np.abs(self.vhalf2())**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def q1(self):
        """Def quadrant 1, subsequent functions do the same for other quadrants """
        q1 = (self.x >= 0)*(self.y >= 0)
        return self.field*q1
    
    def q2(self):
        q2 = (self.x < 0)*(self.y >= 0)
        return self.field*q2
    
    def q3(self):
        q3 = (self.x < 0)*(self.y < 0)
        return self.field*q3
    
    def q4(self):
        q4 = (self.x >= 0)*(self.y < 0)
        return self.field*q4
    
    def get_Power_q1(self):  
        """Get Power of quadrant 1"""                    
        return np.sum(np.abs(self.q1())**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def get_Power_q2(self):
        return np.sum(np.abs(self.q2())**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def get_Power_q3(self):
        return np.sum(np.abs(self.q3())**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def get_Power_q4(self):
        return np.sum(np.abs(self.q4())**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def section(self, ang_min, ang_max):
        """Return field distribution of given section, defined by minimum angle and maximum angle"""
        if ang_min > ang_max:
            ang_min, ang_max = ang_max, ang_min
        if ang_max <= np.pi and ang_min <= np.pi:
            sec = (np.arctan2(self.y,self.x)>=ang_min)*(np.arctan2(self.y,self.x)<ang_max)
        if ang_max> np.pi and ang_min <= np.pi:
            sec1 = (np.arctan2(self.y,self.x)>=ang_min)
            ang_max = ang_max-2*np.pi
            sec2 = (np.arctan2(self.y,self.x)<ang_max)
            sec = sec1 + sec2
        if ang_max>np.pi and ang_min > np.pi:
            ang_max = ang_max - 2*np.pi
            ang_min = ang_min - 2*np.pi
            sec = (np.arctan2(self.y,self.x)>=ang_min)*(np.arctan2(self.y,self.x)<ang_max)

        return self.field*sec
    
    def get_Power_section(self, ang_min, ang_max):
        """Get power of given section"""
        return np.sum(np.abs(self.section(ang_min, ang_max))**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def nsplit_mode(self, N, rot_ang=0):
        """Subdivide xy plane by N equal angles (ang). Let i be index of the subdivision and multiply each subdivision by e^1j*i*ang. """
        d_ang = 2*np.pi/N 
        mask = np.zeros((self.Dy, self.Dx), dtype='complex128')
        for i in range(N):
            ang_min = i*d_ang + rot_ang
            ang_max = (i+1)*d_ang + rot_ang
            if ang_max <= np.pi and ang_min <= np.pi:
                sec = (np.arctan2(self.y,self.x)>=ang_min)*(np.arctan2(self.y,self.x)<ang_max)
            if ang_max> np.pi and ang_min <= np.pi:
                sec1 = (np.arctan2(self.y,self.x)>=ang_min)
                ang_max = ang_max-2*np.pi
                sec2 = (np.arctan2(self.y,self.x)<ang_max)
                sec = sec1 + sec2
            if ang_max>np.pi and ang_min > np.pi:
                ang_max = ang_max - 2*np.pi
                ang_min = ang_min - 2*np.pi
                sec = (np.arctan2(self.y,self.x)>=ang_min)*(np.arctan2(self.y,self.x)<ang_max)
            mask += sec*np.exp(1j*i*d_ang)
        self.field = self.field*mask
        return self
    
    def get_data_nsplit_detector(self, N, rot_angle=0):
        data = []
        d_ang = 2*np.pi/N
        for i in range(int(N/2)):
            pos_measure = self.get_Power_section(i*d_ang + rot_angle, (i+1)*d_ang + rot_angle)
            neg_measure = self.get_Power_section(i*d_ang + np.pi + rot_angle, (i+1)*d_ang + np.pi + rot_angle)
            data.append(pos_measure - neg_measure)
        data = int(N/2)*np.array(data)
        return data
    
    def int_profile(self):
        """ Get intensity profile of the field within Space"""
        return np.abs(self.field)**2
    
    def phase(self, twopi=False):
        """Get phase profile of the field."""
        if twopi == False:
            return np.angle(self.field)
        if twopi == True:
            p = np.angle(self.field)
            neg = p<0
            p = p + neg*2*np.pi
            return p
    
    def norm_beam(self):                                        #normalize beam so that Total Power = sum(field**2 * Delta(x)*Delta(y)) = 1
        self.field = self.field/np.sqrt(self.get_Power())
        return self
    
    def Max_int1(self):                                         #rescale the field such that the maximum Intensity point is equal to one.
        self.field = self.field/np.sqrt(np.amax(np.abs(self.field)**2))
        return self

    def propagate(self, z):                                     #propagate the field by a distance z using fresnel integral with exp(-ikz)
        k = 2*np.pi/self.lamb
        prop = np.exp(1j * z * (self.kx ** 2 + self.ky ** 2) / (2 * k) - 1j*k*z)
        self.fourier_field = fft.fft2(self.field)
        self.field = fft.ifft2(prop*self.fourier_field)
        return self
    

    def lens(self, f, f0=(0,0)):                                #apply a lens operator to the field, with lens center at f0 and focus lenght equal to f
        k = 2*np.pi/self.lamb
        self = self*np.exp(-1j*k*(((self.x-f0[0])**2 + (self.y-f0[1])**2)/(2*f)))
        return self

    def astigmatic_lens(self, fx, fy, f0=(0,0)):                      #apply a astigmatic lens with two focal axis, with focus fx and fy.
        k = 2*np.pi/self.lamb
        self = self*np.exp(-1j*k*(((self.x-f0[0])**2)/fx + ((self.y-f0[1])**2)/fy)/2)
        return self
    
    def tilted_lens(self, f, phi, f0=(0,0)):                      #apply a astigmatic lens with two focal axis, with each focus given by a tilt phi.
        k = 2*np.pi/self.lamb
        self = self*np.exp(-1j*k*(((self.x-f0[0])**2)/np.cos(phi) + ((self.y-f0[1])**2)*np.cos(phi))/(2*f))
        return self
    
    def stripe_v(self, size):
        """Def vertical stripe"""
        s = size*self.Dx/(2*self.nix)
        self.field[:,int(self.Dx/2 - s):int(self.Dx/2+s)] = 0
        return self
    
    def stripe_h(self, size):
        """Def horizontal stripe"""
        s = size*self.Dy/(2*self.niy)
        self.field[int(self.Dy/2 - s):int(self.Dy/2+s),:] = 0
        return self
    
    def stripe_cross(self, size, hsize=None):
        """Def cross stripe"""
        if hsize == None:
            hsize = size
        s = size*self.Dx/(2*self.nix)
        sh = hsize*self.Dy/(2*self.niy)
        self.field[int(self.Dy/2 - sh):int(self.Dy/2+sh),:] = 0
        self.field[:,int(self.Dx/2 - s):int(self.Dx/2+s)] = 0
        return self

    def slit(self, size):
        """Def 1 slit"""
        s = size*self.Dx/(2*self.nix)
        self.field[:,:int(self.Dx/2-s)] = 0
        self.field[:,int(self.Dx/2 + s):] = 0
        return self
    
    def double_slit(self, size, dis):
        """Def double slit"""
        s = size*self.Dx/(2*self.nix)
        d = dis*self.Dx/(2*self.nix)
        self.field[:,0:int((self.Dx-d-s)/2)] = 0
        self.field[:,int((self.Dx-d+s)/2):int((self.Dx+d-s)/2)] = 0
        self.field[:,int((self.Dx+d+s)/2):] = 0
        return self
    
    def apply_mask(self, mask):
        """Applies a mask to the field"""
        self.field = self.field*np.exp(-1j*mask.mask)
        return self
    
    def center_mass(self):
        """Calculates the center of mass of intensities of a given field"""
        return ndimage.center_of_mass(self.int_profile())
    
    def std(self):
        """Calculate the standard deviation of intensities"""
        c = self.center_mass()
        return np.sqrt(np.average((self.x-self.x[0,int(c[1])])**2+(self.y-self.y[int(c[0]),0])**2, weights=self.int_profile()))
    
    def crop(self, center=None, std=None, window=2):
        """Crops a field by its std*window arround the center of mass"""
        if center == None:
            center = self.center_mass()
        if std == None:
            std = self.std()
        xmin = int(center[1] - window*std*self.Dx/self.nix/2)
        xmax = int(center[1] + window*std*self.Dx/self.nix/2)
        ymin = int(center[0] - window*std*self.Dy/self.niy/2)
        ymax = int(center[0] + window*std*self.Dy/self.niy/2)
        self.x = self.x[:,xmin:xmax]
        self.y = self.y[ymin:ymax, :]
        self.field = self.field[ymin:ymax, xmin:xmax]
        self.Dx = len(self.x[0,:])
        self.Dy = len(self.y[:,0])
        self.nix = (self.x[0,-1] - self.x[0,0])/2
        self.niy = (self.y[-1,0] - self.y[0,0])/2
        self.x0 = center[1]
        self.y0 = center[0]
        return self

    



def overlap(b1, b2): # calculate overlap between two beams, only properly works if ni and D of beams are equal.
    return np.abs(np.sum(b1.field*np.conjugate(b2.field))*(4*b1.nix/b1.Dx)*(b1.niy/b1.Dy))**2/(b1.get_Power()*b2.get_Power())