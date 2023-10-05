from typing import Union
from typing import Tuple
from typing import Any
import numpy
import sympy
import scipy
import control

import matplotlib.pyplot as plt


class ControlUtils:

    @staticmethod
    def hinfnorm(sys: Union[control.iosys.LinearIOSystem,control.NonlinearIOSystem,control.TransferFunction],
                 freq=numpy.logspace(0, 8, 8000)) -> numpy.number:
        """_summary_

        Args:
            sys (Union[control.statesp.StateSpace, control.iosys.TransferFunction]): Linear System  
            freq (_type_, optional): Frequency grid over which to find the norm. Defaults to numpy.logspace(0,5,1000).

        Returns:
            _type_: The Hinf norm of the system
        """
        if(sys.outputs > 1 or sys.outputs > 1):
            return numpy.array([numpy.linalg.svd(sys(1.0j*w), compute_uv=False) for w in freq]).max()
        else:
            return numpy.array([numpy.abs(sys(1.0j*w)) for w in freq]).max()
        
    @staticmethod
    def balaced_truncation( sys: Union[control.LinearIOSystem,control.NonlinearIOSystem],
                red_ord: int ) -> Tuple[control.LinearIOSystem, numpy.ndarray, Tuple[Any]] :
        """_summary_

        Args:
            sys (Union[control.NonlinearIOSystem,control.LinearIOSystem]): FOM 
            red_ord (int): Order of the ROM

        Returns:
            Tuple[control.LinearIOSystem, numpy.ndarray]: ROM and Hankel Singular Values
        """
        # Getting Grammians (Factorized, check documentation for control.gram(.,'cf'/'of'))
        WcL = numpy.array( control.gram(sys,'cf'), dtype=numpy.longdouble ).T
        WoL = numpy.array( control.gram(sys,'of'), dtype=numpy.longdouble ).T

        # Getting SVD
        U, S2, V = numpy.linalg.svd(WcL.T @ (WoL @ WoL.T) @ WcL, compute_uv=True)
        S = numpy.diag( S2**(1/2) )

        # Getting transformation
        T = ( (S**(1/2)) @ U.T ) @ numpy.linalg.inv(WcL)
        invT = numpy.linalg.inv(T)

        # Computing balanced Grammians
        WcBal = T @ (WcL @ WcL.T) @ T.T
        WoBal = invT.T @ (WoL @ WoL.T) @ invT
        
        # Getting Hankel singular values
        hsv = numpy.linalg.eigvals(WcBal@WoBal)**(1/2)

        # Creating Projection Matrices
        proj = numpy.row_stack([numpy.eye(red_ord,red_ord), numpy.zeros((sys.A.shape[0]-red_ord,red_ord))])
        V_proj = invT @ proj
        WT_proj = proj.T @ T

        # Get ROM matrices
        A_red = WT_proj @ sys.A @ V_proj
        B_red = WT_proj @ sys.B
        C_red = sys.C @ V_proj

        # Create ROM
        sys_red = control.ss(A_red,B_red,C_red,sys.D)
        return sys_red, hsv, (T,proj,V_proj,WT_proj) # type: ignore


class ODEUtils:

    @staticmethod
    def solveLinearSystem(plant: control.iosys.StateSpace,
                            controller: control.iosys.StateSpace,
                            xp0: numpy.ndarray = numpy.empty(0),
                            xc0: numpy.ndarray = numpy.empty(0),
                            uVec: numpy.ndarray = numpy.empty(0),
                            t_ini=0,
                            t_fin=10,
                            T_step=0.01,
                            print_stats: bool = True
                            ):
        """_summary_

        Args:
            plant (control.iosys.StateSpace): Plant
            discrete_controller (control.iosys.StateSpace): Discrete time controller
            xp0 (numpy.ndarray, optional): Initial state of the plant. Defaults to numpy.array([0]).
            xc0 (numpy.ndarray, optional): Initial state of the discrete time controller. Defaults to numpy.array([0]).
            uVec (numpy.ndarray, optional): Values of the input. Defaults to numpy.array([0]).
            t_ini (int, optional): Initial time. Defaults to 0.
            t_fin (int, optional): Final time. Defaults to 10.
            T_step (float, optional): Simulation step. Defaults to 0.01.
            print_stats (bool, optional): Flag to print integration loop stats. Defaults to True.

        Returns:
            _type_: _description_
        """
        steps = int(numpy.ceil((t_fin-t_ini)/T_step))

        tVec = numpy.linspace(t_ini, t_fin, steps)[:, None]
        uVec = numpy.zeros((steps, 1)) if len(
            uVec) == 0 else uVec.reshape(steps, 1)

        controllerOrder = controller.A.shape[0]  # type: ignore
        Ac = controller.A
        Bv = controller.B
        Cw = controller.C
        Dw = controller.D

        plantOrder = plant.A.shape[0]  # type: ignore
        Ap = plant.A
        Bw = plant.B[:, 0].reshape(plantOrder, 1)  # type: ignore
        Bu = plant.B[:, 1].reshape(plantOrder, 1)  # type: ignore
        Cv = plant.C[0, :].reshape(1, plantOrder)  # type: ignore
        Cy = plant.C[1, :].reshape(1, plantOrder)  # type: ignore
        
        # Closing-loop
        Acl = numpy.row_stack( [ numpy.column_stack( [ Ap + Bw@Dw@Cv, Bw@Cw ] ),
                              numpy.column_stack( [ Bv@Cv, Ac ] )
                            ] )
        Bcl = numpy.row_stack( [ numpy.column_stack([Bw, Bu]),
                                    numpy.column_stack( [ numpy.zeros( (Ac.shape[0],Bw.shape[1]) ), numpy.zeros( (Ac.shape[0],Bu.shape[1]) ) ] )
                                    ] )
        Ccl = numpy.row_stack( [ numpy.column_stack( [ Cv, numpy.zeros( (Cv.shape[0],Ac.shape[0]) ) ] ),
                                    # numpy.column_stack( [ Cv@Ap + Cv@Bw@Dw@Cv, Cv@Bw@Cw ] ),
                                    numpy.column_stack( [ Cy, numpy.zeros( (Cy.shape[0],Ac.shape[0]) ) ] )
                                    ] )
        Dcl = numpy.row_stack( [ numpy.column_stack( [ numpy.zeros( (Cv.shape[0],Bw.shape[1]) ), numpy.zeros( (Cv.shape[0],Bu.shape[1]) ) ] ),
                                    # numpy.column_stack( [ Cv@Bw , Cv@Bu] ),
                                    numpy.column_stack( [ numpy.zeros( (Cy.shape[0],Bw.shape[1]) ), numpy.zeros( (Cy.shape[0],Bu.shape[1]) ) ] )
                                    ] )
        clOrder = Acl.shape[0]
        # mathcalPcl = control.ss( Acl, Bcl, Ccl, Dcl )
        
        xclStart = numpy.row_stack( [xp0, xc0] )
        
        # Execute simulation
        uHandler = scipy.interpolate.interp1d(
            tVec.squeeze(), uVec[:, :].T, kind='linear', copy=False,
            bounds_error=False, fill_value=(uVec[0, :].T, uVec[-1, :].T))
        def dxpHandler(t, xcl): 
            threshold = 1e10
            if not (numpy.linalg.norm(xcl) < 1e10):
                print(f"The norm of the state is out of threshold: {threshold}")
                # raise Exception("The norm of the state is out of threshold")
            return ( (Acl@xcl.reshape(clOrder, 1)) + (Bcl[:,1:2]@uHandler([t])) ).reshape(clOrder,)
        xpclVec = scipy.integrate.solve_ivp(dxpHandler,
                                                [tVec[0, 0], tVec[-1, 0]],
                                                xclStart.reshape(clOrder,),
                                                method='LSODA',
                                                t_eval=tVec.squeeze(),
                                                vectorized=False, atol=1e-4, rtol=1e-4)
        return xpclVec


    @staticmethod
    def solveSDLinearSystem(plant: control.iosys.StateSpace,
                            discrete_controller: control.iosys.StateSpace,
                            xp0: numpy.ndarray = numpy.empty(0),
                            xc0: numpy.ndarray = numpy.empty(0),
                            uVec: numpy.ndarray = numpy.empty(0),
                            t_ini=0,
                            t_fin=10,
                            T_step=0.01,
                            T_samps: numpy.ndarray = numpy.empty(0),
                            print_stats: bool = True
                            ):
        """_summary_

        Args:
            plant (control.iosys.StateSpace): Plant
            discrete_controller (control.iosys.StateSpace): Discrete time controller
            xp0 (numpy.ndarray, optional): Initial state of the plant. Defaults to numpy.array([0]).
            xc0 (numpy.ndarray, optional): Initial state of the discrete time controller. Defaults to numpy.array([0]).
            uVec (numpy.ndarray, optional): Values of the input. Defaults to numpy.array([0]).
            t_ini (int, optional): Initial time. Defaults to 0.
            t_fin (int, optional): Final time. Defaults to 10.
            T_step (float, optional): Simulation step. Defaults to 0.01.
            T_samps (numpy.ndarray, optional): Vector of sampling time instances. Defaults to numpy.array([0]).
            print_stats (bool, optional): Flag to print integration loop stats. Defaults to True.

        Returns:
            _type_: _description_
        """
        steps = int(numpy.ceil((t_fin-t_ini)/T_step))

        tVec = numpy.linspace(t_ini, t_fin, steps)[:, None]
        T_samps = T_samps.reshape(T_samps.size, 1)
        uVec = numpy.zeros((steps, 1)) if len(
            uVec) == 0 else uVec.reshape(steps, 1)

        controllerOrder = discrete_controller.A.shape[0]  # type: ignore
        Ac = discrete_controller.A
        Bv = discrete_controller.B
        Cw = discrete_controller.C
        Dw = discrete_controller.D

        plantOrder = plant.A.shape[0]  # type: ignore
        Ap = plant.A
        Bw = plant.B[:, 0].reshape(plantOrder, 1)  # type: ignore
        Bu = plant.B[:, 1].reshape(plantOrder, 1)  # type: ignore
        Cv = plant.C[0, :].reshape(1, plantOrder)  # type: ignore
        Cy = plant.C[1, :].reshape(1, plantOrder)  # type: ignore

        # Initialization of vectors save simulation
        xpVec = numpy.empty((plantOrder, steps))
        xcVec = numpy.empty((controllerOrder, steps))
        vsVec = numpy.empty((1, steps))
        whVec = numpy.empty((1, steps))

        # Loop variables initialization
        k = 0
        i0 = 0
        i1 = 0

        xpSubStart = numpy.zeros((plantOrder, 1)) if len(
            xp0) == 0 else xp0*numpy.ones((plantOrder, 1))
        xck = numpy.zeros((controllerOrder, 1)) if len(
            xc0) == 0 else xc0*numpy.ones((controllerOrder, 1))
        while k+1 <= T_samps.shape[0]:
            # Final index of the interval
            if k+1 < T_samps.shape[0]:
                i1 = numpy.where(tVec[i0:, 0] > T_samps[k+1, 0])[0][0] + i0
            else:
                i1 = tVec.shape[0]

            # Printing loop params
            if print_stats:
                print(f'idxs: [{i0:5d}, {i1-1:5d}],  ' +
                      f'time interval: [{tVec[i0,0]:6.2f}, {tVec[i1-1,0]:5.2f}]',
                      end='\r',
                      flush=True)

            # Plant input values for the interval
            vs = Cv@xpSubStart
            wh = (Cw@xck) + (Dw@vs)

            # Save controller input, output, and state
            vsVec[:, i0:i1] = numpy.repeat(vs, i1-i0, 1)
            whVec[:, i0:i1] = numpy.repeat(wh, i1-i0, 1)
            xcVec[:, i0:i1] = numpy.repeat(xck, i1-i0, 1)

            # Execute simulation for current interval (i.e. find solution with ODE solver for current interval)
            uHandler = scipy.interpolate.interp1d(
                tVec[i0:i1, 0], uVec[i0:i1, :].T, kind='linear', copy=False,
                bounds_error=False, fill_value=(uVec[i0, :].T, uVec[i1-1, :].T))
            def dxpHandler(t, xp): return (
                (Ap@xp.reshape(plantOrder, 1)) + (Bu@uHandler([t])) + (Bw@wh)).reshape(plantOrder,)
            xpSubVec = scipy.integrate.solve_ivp(dxpHandler,
                                                 [tVec[i0, 0], tVec[i1-1, 0]],
                                                 xpSubStart.reshape(plantOrder,),
                                                 method='RK45',
                                                 t_eval=tVec[i0:i1, 0],
                                                 vectorized=False, atol=1e-5, rtol=1e-5)
            # Save plant state
            xpVec[:, i0:i1] = xpSubVec.y

            # Update controller state
            xck = Ac@xck + Bv@vs

            # Update loop variables
            xpSubStart = xpSubVec.y[:, -1].reshape(plantOrder, 1)
            k = k + 1
            i0 = i1 - 1

        return xpVec, xcVec, vsVec, whVec


    @staticmethod
    def plot_inputs(tVec, tLims, uVec=None, whVec=None):
        idxLims = range(numpy.where(tVec[:, 0] >= tLims[0])[
                        0][0], numpy.where(tVec[:, 0] >= tLims[1])[0][0] + 1)
        if uVec is not None:
            plt.plot(tVec[idxLims, 0], uVec[idxLims, 0], label="$u$")
        if whVec is not None:
            plt.plot(tVec[idxLims, 0], whVec[0, idxLims], label="$w_h$")
        plt.title('Plant inputs')
        plt.legend()
        plt.grid()
        plt.gcf().set_figwidth(10)
        plt.show()

    @staticmethod
    def plot_outputs(tVec, tLims, yVec=None, vVec=None, vsVec=None):
        idxLims = range(numpy.where(tVec[:, 0] >= tLims[0])[
                        0][0], numpy.where(tVec[:, 0] >= tLims[1])[0][0] + 1)
        if yVec is not None:
            plt.plot(tVec[idxLims, 0], yVec[0, idxLims], label="$y$")
        if vVec is not None:
            plt.plot(tVec[idxLims, 0], vVec[0, idxLims], label="$v$")
        if vsVec is not None:
            plt.plot(tVec[idxLims, 0], vsVec[0, idxLims],
                     label="$\\mathcal{H}(v_s)$")
        plt.title('Plant outputs')
        plt.legend()
        plt.grid()
        plt.gcf().set_figwidth(10)
        plt.show()

    @staticmethod
    def plot_controller_io(tVec, tLims, vsVec=None, whVec=None):
        idxLims = range(numpy.where(tVec[:, 0] >= tLims[0])[
                        0][0], numpy.where(tVec[:, 0] >= tLims[1])[0][0] + 1)
        if vsVec is not None:
            plt.plot(tVec[idxLims, 0], vsVec[0, idxLims],
                     label="$\\mathcal{H}(v_s)$")
        if whVec is not None:
            plt.plot(tVec[idxLims, 0], whVec[0, idxLims], label="$w_h$")
        plt.title('Controller input-output')
        plt.legend()
        plt.grid()
        plt.gcf().set_figwidth(10)
        plt.show()
