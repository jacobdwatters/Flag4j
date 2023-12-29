package com.flag4j.sparse_matrix;

import com.flag4j.CooMatrix;
import com.flag4j.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;


class CooMatrixSetRowTests {

    @Test
    void setRowTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        double[] bEntries;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.43471, 0.92285, 0.37443};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{4, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.71106, 0.44913, 0.42535, 0.75227, 0.9498};

        expShape = new Shape(3, 5);
        expEntries = new double[]{0.71106, 0.44913, 0.42535, 0.75227, 0.9498, 0.92285, 0.37443};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 1, 2};
        expColIndices = new int[]{0, 1, 2, 3, 4, 2, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(bEntries, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.20433, 0.43246, 0.70123, 0.89993, 0.8816};
        aRowIndices = new int[]{6, 9, 11, 15, 20};
        aColIndices = new int[]{8, 6, 9, 10, 6};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.44274, 0.93158, 0.2449, 0.09041, 0.41, 0.9001, 0.08126, 0.43374, 0.10697, 0.02379, 0.37932};

        expShape = new Shape(23, 11);
        expEntries = new double[]{0.20433, 0.43246, 0.70123, 0.89993, 0.44274, 0.93158, 0.2449, 0.09041, 0.41, 0.9001, 0.08126, 0.43374, 0.10697, 0.02379, 0.37932, 0.8816};
        expRowIndices = new int[]{6, 9, 11, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 20};
        expColIndices = new int[]{8, 6, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(bEntries, 16));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new double[]{0.41034, 0.59522, 0.28929, 0.96894, 0.91679, 0.28692, 0.13537, 0.66813, 0.98182};
        aRowIndices = new int[]{3, 159, 242, 523, 561, 576, 621, 653, 908};
        aColIndices = new int[]{4, 1, 3, 4, 4, 4, 4, 1, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.77537, 0.41015, 0.11228, 0.66719, 0.86707};

        expShape = new Shape(1000, 5);
        expEntries = new double[]{0.41034, 0.59522, 0.28929, 0.96894, 0.91679, 0.28692, 0.13537, 0.66813, 0.98182, 0.77537, 0.41015, 0.11228, 0.66719, 0.86707};
        expRowIndices = new int[]{3, 159, 242, 523, 561, 576, 621, 653, 908, 999, 999, 999, 999, 999};
        expColIndices = new int[]{4, 1, 3, 4, 4, 4, 4, 1, 0, 0, 1, 2, 3, 4};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(bEntries, 999));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.62926, 0.39912, 0.97867, 0.05437};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{0, 0, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.88629, 0.7154, 0.23673};

        CooMatrix final0a = a;
        double[] final0b = bEntries;
        assertThrows(Exception.class, ()->final0a.setRow(final0b, 6));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.06788, 0.11886, 0.18054, 0.24061};
        aRowIndices = new int[]{1, 3, 3, 4};
        aColIndices = new int[]{1, 0, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.63856, 0.89354, 0.0177, 0.28224, 0.40056};

        CooMatrix final1a = a;
        double[] final1b = bEntries;
        assertThrows(Exception.class, ()->final1a.setRow(final1b, 3));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.72158, 0.85906, 0.82353, 0.86208};
        aRowIndices = new int[]{1, 1, 2, 3};
        aColIndices = new int[]{1, 2, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.17677, 0.42461};

        CooMatrix final2a = a;
        double[] final2b = bEntries;
        assertThrows(Exception.class, ()->final2a.setRow(final2b, 3));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.7291, 0.11502, 0.04717, 0.91673};
        aRowIndices = new int[]{0, 1, 2, 4};
        aColIndices = new int[]{1, 1, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[]{0.82171, 0.45353, 0.5833};

        CooMatrix final3a = a;
        double[] final3b = bEntries;
        assertThrows(Exception.class, ()->final3a.setRow(final3b, 19));
    }


    @Test
    void setRowBoxedDoubleTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        double[] aEntries;
        CooMatrix a;

        Double[] bEntries;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        double[] expEntries;
        CooMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new double[]{0.43471, 0.92285, 0.37443};
        aRowIndices = new int[]{0, 1, 2};
        aColIndices = new int[]{4, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Double[]{0.71106, 0.44913, 0.42535, 0.75227, 0.9498};

        expShape = new Shape(3, 5);
        expEntries = new double[]{0.71106, 0.44913, 0.42535, 0.75227, 0.9498, 0.92285, 0.37443};
        expRowIndices = new int[]{0, 0, 0, 0, 0, 1, 2};
        expColIndices = new int[]{0, 1, 2, 3, 4, 2, 0};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(bEntries, 0));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new double[]{0.20433, 0.43246, 0.70123, 0.89993, 0.8816};
        aRowIndices = new int[]{6, 9, 11, 15, 20};
        aColIndices = new int[]{8, 6, 9, 10, 6};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Double[]{0.44274, 0.93158, 0.2449, 0.09041, 0.41, 0.9001, 0.08126, 0.43374, 0.10697, 0.02379, 0.37932};

        expShape = new Shape(23, 11);
        expEntries = new double[]{0.20433, 0.43246, 0.70123, 0.89993, 0.44274, 0.93158, 0.2449, 0.09041, 0.41, 0.9001, 0.08126, 0.43374, 0.10697, 0.02379, 0.37932, 0.8816};
        expRowIndices = new int[]{6, 9, 11, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 20};
        expColIndices = new int[]{8, 6, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 6};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(bEntries, 16));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 5);
        aEntries = new double[]{0.41034, 0.59522, 0.28929, 0.96894, 0.91679, 0.28692, 0.13537, 0.66813, 0.98182};
        aRowIndices = new int[]{3, 159, 242, 523, 561, 576, 621, 653, 908};
        aColIndices = new int[]{4, 1, 3, 4, 4, 4, 4, 1, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Double[]{0.77537, 0.41015, 0.11228, 0.66719, 0.86707};

        expShape = new Shape(1000, 5);
        expEntries = new double[]{0.41034, 0.59522, 0.28929, 0.96894, 0.91679, 0.28692, 0.13537, 0.66813, 0.98182, 0.77537, 0.41015, 0.11228, 0.66719, 0.86707};
        expRowIndices = new int[]{3, 159, 242, 523, 561, 576, 621, 653, 908, 999, 999, 999, 999, 999};
        expColIndices = new int[]{4, 1, 3, 4, 4, 4, 4, 1, 0, 0, 1, 2, 3, 4};
        exp = new CooMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.setRow(bEntries, 999));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.62926, 0.39912, 0.97867, 0.05437};
        aRowIndices = new int[]{0, 1, 2, 2};
        aColIndices = new int[]{0, 0, 0, 1};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Double[]{0.88629, 0.7154, 0.23673};

        CooMatrix final0a = a;
        Double[] final0b = bEntries;
        assertThrows(Exception.class, ()->final0a.setRow(final0b, 6));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.06788, 0.11886, 0.18054, 0.24061};
        aRowIndices = new int[]{1, 3, 3, 4};
        aColIndices = new int[]{1, 0, 2, 0};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Double[]{0.63856, 0.89354, 0.0177, 0.28224, 0.40056};

        CooMatrix final1a = a;
        Double[] final1b = bEntries;
        assertThrows(Exception.class, ()->final1a.setRow(final1b, 3));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.72158, 0.85906, 0.82353, 0.86208};
        aRowIndices = new int[]{1, 1, 2, 3};
        aColIndices = new int[]{1, 2, 2, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Double[]{0.17677, 0.42461};

        CooMatrix final2a = a;
        Double[] final2b = bEntries;
        assertThrows(Exception.class, ()->final2a.setRow(final2b, 3));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new double[]{0.7291, 0.11502, 0.04717, 0.91673};
        aRowIndices = new int[]{0, 1, 2, 4};
        aColIndices = new int[]{1, 1, 1, 2};
        a = new CooMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Double[]{0.82171, 0.45353, 0.5833};

        CooMatrix final3a = a;
        Double[] final3b = bEntries;
        assertThrows(Exception.class, ()->final3a.setRow(final3b, 19));
    }
}
