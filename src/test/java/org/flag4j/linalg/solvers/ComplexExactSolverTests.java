package org.flag4j.linalg.solvers;


import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.CVector;
import org.flag4j.linalg.solvers.exact.ComplexExactSolver;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ComplexExactSolverTests {

    static CNumber[][] aEntries;
    static CNumber[] bEntries;
    static CNumber[] expEntries;

    static CMatrix A;
    static CVector b;
    static CVector exp;

    private static void setMatrices() {
        A = new CMatrix(aEntries);
        b = new CVector(bEntries);
        exp = new CVector(expEntries);
    }


    @Test
    void solveTestCase() {
        ComplexExactSolver solver = new ComplexExactSolver();

        // ----------------- Sub-case 1 -----------------
        aEntries = new CNumber[][]{
                {new CNumber(24.5, -0.35), new CNumber(25.6, -22.04), new CNumber(9.52, 2.3)},
                {new CNumber(-2.04), new CNumber(), new CNumber(5.662)},
                {new CNumber(0.3456), new CNumber(0, -1.56), new CNumber(-23.5, -20.245)}
        };
        bEntries = new CNumber[]{new CNumber(35.6, -6.3), new CNumber(-0.0245, 0.024), new CNumber(0, 100.3)};
        expEntries = new CNumber[]{
                new CNumber("-5.259018068549946 - 7.623306153823768i"),
                new CNumber("-0.025597320154231645 + 8.146170435212893i"),
                new CNumber("-1.899134026817714 - 2.74241337933601i")};
        setMatrices();

        assertEquals(exp, solver.solve(A, b));

        // ----------------- Sub-case 2 -----------------
        aEntries = new CNumber[][]{
                {new CNumber(24.5, -0.35), new CNumber(25.6, -22.04), new CNumber(9.52, 2.3)},
                {new CNumber(-2.04), new CNumber(), new CNumber(5.662)},
                {new CNumber(0.3456), new CNumber(0, -1.56), new CNumber(-23.5, -20.245)}
        };
        bEntries = new CNumber[]{new CNumber(35.6, -6.3), new CNumber(-0.0245, 0.024),
                new CNumber(0, 100.3), new CNumber(94,52)};
        setMatrices();

        assertThrows(IllegalArgumentException.class, ()->solver.solve(A, b));

        // ----------------- Sub-case 3 -----------------
        // TODO: This is not actually considered singular within the threshold. Its determinant is very small but not small enough.
//        aEntries = new CNumber[][]{
//                {new CNumber(24.5, -9.351), new CNumber(-2.56, 99.52)},
//                {new CNumber(-0.38037723083219255, 0.15548985842976704), new CNumber(0.00355, -1.56)}
//        }; // A singular matrix.
//        bEntries = new CNumber[]{new CNumber(35.6, -6.3), new CNumber(-0.0245, 0.024)};
//        expEntries = new CNumber[]{};
//        setMatrices();
//
//        assertThrows(SingularMatrixException.class, ()->solver.solve(A, b));
    }
}
