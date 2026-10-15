import React, { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { base44 } from "@/api/base44Client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Lock, Loader2, AlertTriangle, ShieldCheck } from "lucide-react";
import AuthLayout from "@/components/AuthLayout";

export default function ResetPassword() {
  const [searchParams] = useSearchParams();
  const resetToken = searchParams.get("token");

  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (newPassword !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    setLoading(true);
    try {
      await base44.auth.resetPassword({ resetToken, newPassword });
      window.location.href = "/login";
    } catch (err) {
      setError(err.message || "Failed to reset password");
    } finally {
      setLoading(false);
    }
  };

  if (!resetToken) {
    return (
      <AuthLayout
        icon={AlertTriangle}
        title="Invalid Reset Link"
        subtitle="This password reset link is missing or invalid"
        footer={
          <Link to="/forgot-password" className="text-[#2ED88A] font-bold hover:underline">
            Request a new link
          </Link>
        }
      >
        <p className="text-xs text-[#7A838D] text-center leading-relaxed">
          The password reset token appears to be incomplete or expired. Please request a new password reset email to continue.
        </p>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      icon={ShieldCheck}
      title="New Password"
      subtitle="Enter and confirm your new secure password"
    >
      {error && (
        <div className="mb-4 p-3.5 rounded-xl border border-[#FF5252]/30 bg-[#FF5252]/[0.08] text-[#FF8A8A] text-xs font-medium">
          {error}
        </div>
      )}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-1.5">
          <Label htmlFor="password" className="text-xs font-bold uppercase tracking-wider text-[#7A838D]">New Password</Label>
          <div className="relative group">
            <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[#7A838D] group-focus-within:text-[#2ED88A] transition-colors" aria-hidden="true" />
            <Input
              id="password"
              type="password"
              autoComplete="new-password"
              autoFocus
              placeholder="••••••••"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="pl-10 h-12 bg-[#0F1216] border-[#1E232A] text-[#F2F5F7] placeholder:text-[#4A525A] focus-visible:ring-1 focus-visible:ring-[#2ED88A] focus-visible:border-[#2ED88A] rounded-xl transition-all"
              required
            />
          </div>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="confirm" className="text-xs font-bold uppercase tracking-wider text-[#7A838D]">Confirm New Password</Label>
          <div className="relative group">
            <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[#7A838D] group-focus-within:text-[#2ED88A] transition-colors" aria-hidden="true" />
            <Input
              id="confirm"
              type="password"
              autoComplete="new-password"
              placeholder="••••••••"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="pl-10 h-12 bg-[#0F1216] border-[#1E232A] text-[#F2F5F7] placeholder:text-[#4A525A] focus-visible:ring-1 focus-visible:ring-[#2ED88A] focus-visible:border-[#2ED88A] rounded-xl transition-all"
              required
            />
          </div>
        </div>

        <Button 
          type="submit" 
          className="w-full h-12 font-bold tracking-wide uppercase text-xs bg-[#2ED88A] text-[#07080A] hover:bg-[#38ef9b] hover:shadow-[0_0_20px_rgba(46,216,138,0.35)] active:scale-[0.98] rounded-xl transition-all mt-2" 
          disabled={loading}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Resetting...
            </span>
          ) : (
            "Update Password"
          )}
        </Button>
      </form>
    </AuthLayout>
  );
}
