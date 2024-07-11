package org.flag4j.vector;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CVector;
import org.flag4j.dense.Vector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorCrossProductTests {

    double[] aEntries = {1.0, 5.6, -9.355};
    Vector a = new Vector(aEntries);

    @Test
    void realCrossTestCase() {
        double[] bEntries, expEntries;
        Vector b, exp;

        // --------------------- Sub-case 1 ---------------------
        bEntries = new double[]{45.6, 7.9, -0.2345};
        b = new Vector(bEntries);
        expEntries = new double[]{72.59130000000002, -426.3535, -247.45999999999998};
        exp = new Vector(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- Sub-case 2 ---------------------
        bEntries = new double[]{0, 0, 0};
        b = new Vector(bEntries);
        expEntries = new double[]{0.0, -0.0, 0.0};
        exp = new Vector(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- Sub-case 3 ---------------------
        bEntries = new double[]{0, 0, 1};
        b = new Vector(bEntries);
        expEntries = new double[]{5.6, -1.0, 0.0};
        exp = new Vector(expEntries);

        assertEquals(exp, a.cross(b));

        // --------------------- Sub-case 4 ---------------------
        bEntries = new double[]{3, 1, Double.POSITIVE_INFINITY};
        b = new Vector(bEntries);
        expEntries = new double[]{Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, -15.799999999999997};
        exp = new Vector(expEntries);

        assertEquals(exp, a.cross(b));
    }

    @Test
    void complexCrossTestCase() {
        CNumber[] bEntries, expEntries;
        CVector b, exp;

        // --------------------- Sub-case 1 ---------------------
        bEntries = new CNumber[]{new CNumber("1.55+87.1i"), new CNumber("-0.00234-8.0i"), new CNumber("0.0+54.2i")};
        b = new CVector(bEntries);
        expEntries = new CNumber[]{new CNumber("-0.021890700000000003+228.67999999999998i"),
                new CNumber("-14.500250000000001-869.0205000000001i"),
                new CNumber("-8.68234-495.75999999999993i")};
        exp = new CVector(expEntries);

        assertEquals(exp, a.cross(b));
    }
}
