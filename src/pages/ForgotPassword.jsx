import React, { useState } from "react";
import { Link } from "react-router-dom";
import { base44 } from "@/api/base44Client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Mail, ArrowLeft, Loader2, KeyRound } from "lucide-react";
import AuthLayout from "@/components/AuthLayout";

export default function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      await base44.auth.resetPasswordRequest(email);
    } catch {
      // Always show success regardless
    } finally {
      setLoading(false);
      setSent(true);
    }
  };

  return (
    <AuthLayout
      icon={KeyRound}
      title="Reset Password"
      subtitle="Enter your email to receive a secure password recovery link"
      footer={
        <Link 
          to="/login" 
          className="text-[#2ED88A] font-semibold hover:text-[#38ef9b] transition-colors inline-flex items-center gap-1.5 group"
        >
          <ArrowLeft className="w-3.5 h-3.5 transition-transform group-hover:-translate-x-1" />
          <span>Back to log in</span>
        </Link>
      }
    >
      {sent ? (
        <div className="p-4 rounded-xl border border-[#2ED88A]/30 bg-[#2ED88A]/[0.05] text-center space-y-2 animate-in fade-in zoom-in-95 duration-300">
          <div className="w-10 h-10 mx-auto rounded-full bg-[#2ED88A]/10 border border-[#2ED88A]/30 flex items-center justify-center text-[#2ED88A]">
            <Mail className="w-5 h-5" />
          </div>
          <p className="text-sm text-[#F2F5F7] font-medium">Check your inbox</p>
          <p className="text-xs text-[#7A838D] leading-relaxed">
            If an account exists with <span className="text-[#2ED88A] font-mono">{email}</span>, you will receive a password reset link shortly.
          </p>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-2">
            <Label htmlFor="email" className="text-xs font-bold uppercase tracking-wider text-[#7A838D]">
              Email Address
            </Label>
            <div className="relative group">
              <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[#7A838D] group-focus-within:text-[#2ED88A] transition-colors" aria-hidden="true" />
              <Input
                id="email"
                type="email"
                autoComplete="email"
                autoFocus
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="pl-10 h-12 bg-[#0F1216] border-[#1E232A] text-[#F2F5F7] placeholder:text-[#4A525A] focus-visible:ring-1 focus-visible:ring-[#2ED88A] focus-visible:border-[#2ED88A] rounded-xl transition-all"
                required
              />
            </div>
          </div>
          <Button 
            type="submit" 
            className="w-full h-12 font-bold tracking-wide uppercase text-xs bg-[#2ED88A] text-[#07080A] hover:bg-[#38ef9b] hover:shadow-[0_0_20px_rgba(46,216,138,0.35)] active:scale-[0.98] rounded-xl transition-all duration-200" 
            disabled={loading}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Sending Link...
              </span>
            ) : (
              "Send Reset Link"
            )}
          </Button>
        </form>
      )}
    </AuthLayout>
  );
}
