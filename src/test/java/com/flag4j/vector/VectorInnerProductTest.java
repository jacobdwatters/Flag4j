package com.flag4j.vector;

import com.flag4j.CVector;
import com.flag4j.CooCVector;
import com.flag4j.CooVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorInnerProductTest {

    int[] indices;
    int sparseSize;
    double[] aEntries = {1.0, 5.6, -9.355, 215.0};
    Vector a = new Vector(aEntries);

    @Test
    void realDenseTestCase() {
        double[] bEntries;
        Vector b;
        Double exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{-0.2344, 0.0, 1.4667, 712.45};
        b = new Vector(bEntries);
        exp = 153162.7946215;

        assertEquals(exp, a.inner(b));
    }


    @Test
    void realSparseTestCase() {
        double[] bEntries;
        CooVector b;
        Double exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new double[]{1.4667};
        indices = new int[]{2};
        sparseSize = 4;
        b = new CooVector(sparseSize, bEntries, indices);
        exp = -13.7209785;

        assertEquals(exp, a.inner(b));
    }


    @Test
    void complexDenseTestCase() {
        CNumber[] bEntries;
        CVector b;
        CNumber exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{new CNumber("1.45-9.8i"), new CNumber("0.0+98.234i"),
                new CNumber("0.134"), new CNumber("-99.24+45.008i")};
        b = new CVector(bEntries);
        exp = new CNumber("-21336.40357-10217.030400000001j");

        assertEquals(exp, a.inner(b));
    }


    @Test
    void complexSparseTestCase() {
        CNumber[] bEntries;
        CooCVector b;
        CNumber exp;

        // ----------------------- Sub-case 1 -----------------------
        bEntries = new CNumber[]{new CNumber("1.45-9.8i"), new CNumber("-99.24+45.008i")};
        indices = new int[]{0, 3};
        sparseSize = 4;
        b = new CooCVector(sparseSize, bEntries, indices);
        exp = new CNumber("-21335.149999999998-9666.920000000002j");

        assertEquals(exp, a.inner(b));
    }


    @Test
    void normalizeTestCase() {
        double[] expEntries;
        Vector exp;

        // ----------------------- Sub-case 1 -----------------------
        expEntries = new double[]{0.0046451435284722955, 0.026012803759444855, -0.043455317708858326, 0.9987058586215436};
        exp = new Vector(expEntries);

        assertEquals(exp, a.normalize());
    }
}
