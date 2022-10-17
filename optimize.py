from ortools.graph.python import min_cost_flow
import numpy as np
import pandas as pd
import pickle

# Optimizes assignment of TAs to courses.
#
# Students identify how many courses they can TA, and mark which courses they'd like to TA. Instructors identify how many TA spaces they have, and mark which TAs they'd prefer. Preferences are on a five point scale: [most wanted, wanted, somewhat wanted, willing, not at all]
#
# There are three phases: first PhDs are assigned, then MScs, then others. This takes into account priorities for TAships
#
# No assignments are made where there isn't at least willingness (by instructors and TAs) for the assignment
#
# The status quo can be run by having instructors and TAs mark each other as 'most wanted'
#
# Other solutions:
# https://github.com/priyeshjain94/Teaching-Assistant-Assignment-Problem

def makeRandData(fn = 'config.pkl'):
    """
    makeRandData

    This makes a random configuration with the number of classes, PhDs, MScs, and other TAs, the number of spots per course, the available spaces of TAs, and preferences mapping students to courses and vice versa. It saves it to a pickle.

    Input:
    fn: output filename, as text

    """
    
    config = {}
    config['numClasses'] = 100
    config['numPhDs'] = 45
    config['numMSCs'] = 65
    config['numOther'] = 30
    totTAs = config['numPhDs']+config['numMSCs']+config['numOther']

    # courses have between 2 and 8 TAs, assume Gaussian
    spots = np.rint(np.random.normal(loc= 5, scale = 2, size = config['numClasses']))
    spots[ spots>8] = 8
    spots[ spots<2] = 2
    config['spots'] = np.int_(spots)

    # students can teach 1 or 2 classes. Make this binomial maybe?
    config['avail'] = np.int_(np.random.randint(low=1, high=3, size = totTAs))
    
    # students provide preferences: 1s, 2s, 3s, willings (4s).
    # Costs are: 0, 25, 50, 100
    # Anything else is infinite cost
    # format is [[0..numCourses][0..numCourses]...numStudents]
    #studentPrefs = np.round(np.random.gamma( shape = 2, scale = 0.5, size = totTAs*config['numClasses']))
    props = np.asarray([0.05, 0.15, 0.2, 0.4, 0.2])
    costs = [0, 25, 50, 100, np.iinfo(np.int32).max /2 -1]
    indices = np.random.choice( np.prod( props.shape ), size=config['numClasses']*totTAs, p=props.ravel())
    config['studentPrefs'] = list(map(lambda i: costs[i], indices))

    # courses provide preferences: 1s, 2s, 3s, willings (4s).
    # Costs are: 0, 25, 50, 100
    # Anything else is infinite cost
    # format is [[0..numCourses][0..numCourses]...numStudents]
    props = np.asarray([0.01, 0.19, 0.3, 0.4, 0.1])
    costs = [0, 25, 50, 100, np.iinfo(np.int32).max /2 -1]
    indices = np.random.choice( np.prod( props.shape ), size=config['numClasses']*totTAs, p=props.ravel())
    config['coursePrefs'] = list(map(lambda i: costs[i], indices))

    # write config
    with open( fn, 'wb') as f:
        pickle.dump( config, f )

def solve(fn='config.pkl'):
    """
    solve

    Create a simple min-cost max-flow graph, solve it

    Input
    -----
    fn : str
      filename of configuration file
    """
    
    config = pd.read_pickle( fn )

    numTAs = [ config['numPhDs'], config['numMSCs'], config['numOther'] ]
    numClasses = config['numClasses']

    iStart = 0
    spots = config['spots']
    for nGroup in numTAs:
        avail = config['avail'][iStart:(iStart+nGroup)]
        studentPrefs = config['studentPrefs'][ iStart*numClasses:(iStart+nGroup)*numClasses]
        coursePrefs = config['coursePrefs'][ iStart*numClasses:(iStart+nGroup)*numClasses]
        
        smcf = min_cost_flow.SimpleMinCostFlow()

        for iStudent in range(nGroup):
            for iClass in range(numClasses):
                totalCost = studentPrefs[iStudent*nGroup+iClass] + coursePrefs[iStudent*nGroup+iClass]
                if totalCost < 1000:
                    smcf.add_arc_with_capacity_and_unit_cost( iStudent, iClass+nGroup, 1, totalCost )
            smcf.set_node_supply(iStudent, avail[iStudent])

        for iClass in range(numClasses):
            smcf.set_node_supply(iClass+nGroup, -1*spots[iClass])
            
        status = smcf.solve_max_flow_with_min_cost()
        
        if status == smcf.OPTIMAL:
            print('Total cost = ', smcf.optimal_cost())
            print()
            for arc in range(smcf.num_arcs()):
                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.
                if smcf.flow(arc) > 0:
                    # subtract assignment from pool
                    iClass = smcf.head(arc) - nGroup
                    spots[iClass] -= 1

                    # subtract availability (for reporting)
                    avail[smcf.tail(arc)] -= 1
                    print('TA %d assigned to course %d.  Cost = %d' %
                          (smcf.tail(arc), iClass, smcf.unit_cost(arc)))
            #report        
            for iStudent in range(nGroup):
                if avail[iStudent] > 0:
                    print('TA %d has %d more capacity' % (iStudent, smcf.supply( iStudent ) ) )
            
        else:
            print('There was an issue with the min cost flow input.')
            print(f'Status: {status}')
            

        # output courses with missing slots
        print( spots )
            
        iStart += nGroup
        
    
        
if __name__ == '__main__':
    makeRandData()
    solve()
