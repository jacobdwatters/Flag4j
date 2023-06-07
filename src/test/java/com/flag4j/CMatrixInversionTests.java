package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixInversionTests {
    CNumber[][] aEntries, expEntries;
    CMatrix A, exp;


    @Test
    void invTest() {
        // --------------------- Sub-case 1 ---------------------
        aEntries = new CNumber[][]{
                {new CNumber(2.4, 1), new CNumber(9), new CNumber(0, 4)},
                {new CNumber(34), new CNumber(6, 2), new CNumber(24, 7)},
                {new CNumber(6), new CNumber(25, -14.3), new CNumber(14, 43)}
        };
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(-0.020890310281788768, -0.08401682835191364), new CNumber(0.03041740044745477, 0.00290515089267757), new CNumber(-0.011345283148180457, 0.020625927995890685)},
                {new CNumber(0.1502148762820325, 0.01519693661063067), new CNumber(-0.00881122784258471, -0.004051902797816908), new CNumber(-0.007622836636394525, -0.008153804836961875)},
                {new CNumber(0.021439567207676365, 0.09645349255347833), new CNumber(-2.3065935238219268E-4, -0.002301110100512131), new CNumber(0.008806824872674934, -0.029115034319768644)}
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

        assertThrows(IllegalArgumentException.class, ()->A.inv());
        assertThrows(IllegalArgumentException.class, ()->A.T().inv());
    }
}
