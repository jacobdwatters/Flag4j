package org.flag4j.complex_matrix;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixInversionTests {
    CNumber[][] aEntries, expEntries;
    CMatrix A, exp;


    @Test
    void invTestCase() {
        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(2.4, 1), new CNumber(9), new CNumber(0, 4)},
                {new CNumber(34), new CNumber(6, 2), new CNumber(24, 7)},
                {new CNumber(6), new CNumber(25, -14.3), new CNumber(14, 43)}
        };
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(-0.020890310281788768, -0.08401682835191364), new CNumber(0.030417400447454772, 0.00290515089267757), new CNumber(-0.011345283148180455, 0.020625927995890685)},
                {new CNumber(0.1502148762820325, 0.01519693661063067), new CNumber(-0.008811227842584712, -0.004051902797816908), new CNumber(-0.007622836636394531, -0.008153804836961882)},
                {new CNumber(0.021439567207676365, 0.09645349255347833), new CNumber(-2.306593523821924E-4, -0.002301110100512132), new CNumber(0.008806824872674935, -0.029115034319768644)}
        };
        exp = new CMatrix(expEntries);
        assertEquals(exp, A.inv());

        // --------------------- Sub-case 2 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(1), new CNumber(2)},
                {new CNumber(-2), new CNumber(-4)}
        }; // This matrix is singular.
        A = new CMatrix(aEntries);

        assertThrows(RuntimeException.class, ()->A.inv());


        // --------------------- Sub-case 3 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(2.4, 1), new CNumber(9), new CNumber(0, 4)},
                {new CNumber(34), new CNumber(6, 2), new CNumber(24, 7)}
        };
        A = new CMatrix(aEntries);

        assertThrows(LinearAlgebraException.class, ()->A.inv());
        assertThrows(LinearAlgebraException.class, ()->A.T().inv());
    }
}
