"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Settings } from "lucide-react";

const PatientSettings = () => {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-primary">Your Settings</h1>
      <p className="text-lg text-muted-foreground">
        Manage your personal preferences and application settings.
      </p>

      <div className="grid gap-4 md:grid-cols-1">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-xl font-medium flex items-center gap-2">
              <Settings className="h-6 w-6" /> Account Preferences
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-lg">
              <span className="font-semibold">Language:</span> English
            </p>
            <p className="text-lg">
              <span className="font-semibold">Notification Sounds:</span> On
            </p>
            <p className="text-sm text-muted-foreground">
              Options to change password, email, etc. will be here.
            </p>
          </CardContent>
        </Card>

        <div className="p-4 border rounded-lg bg-card">
          <p className="text-muted-foreground">
            More accessibility and display settings will be added here.
          </p>
        </div>
      </div>
    </div>
  );
};

export default PatientSettings;